from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sqlite3

import pytest

from daita._json import FrozenJsonObject, canonical_json
from daita.capabilities import AccessMode, RiskLevel
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
from daita.loop.models import LoopBudgets, LoopPhase, LoopState, Turn
from daita.operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from daita.operations.leases import TaskLease
from daita.operations.models import (
    ActionValidationFacts,
    AgentTrigger,
    Operation,
    OperationStatus,
    Task,
    TaskDependency,
    TaskExecutionFacts,
    TaskStatus,
    TriggerKind,
)
from daita.storage.sqlite import SQLiteCorruptionError, SQLiteOperationStore

NOW = datetime(2026, 7, 17, 19, 0, tzinfo=timezone.utc)
OPERATION_ID = "operation-task-projection"
TRIGGER_ID = "trigger-task-projection"
AGENT_ID = "agent-task-projection"
SESSION_ID = "session-task-projection"
TURN_ID = "turn-task-projection"
MODEL_CALL_ID = "model-call-task-projection"


def _sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _task(
    task_id: str,
    call_id: str,
    *,
    fingerprint_character: str,
    status: TaskStatus,
    attempt: int,
    access_mode: AccessMode = AccessMode.READ,
    risk: RiskLevel = RiskLevel.LOW,
    side_effecting: bool = False,
    idempotent: bool = True,
    replay_safe: bool = True,
    idempotency_key: str | None = None,
    validation_facts: ActionValidationFacts | None = None,
    manual_recovery_reason: str | None = None,
    timestamp: datetime = NOW + timedelta(seconds=1),
) -> Task:
    arguments = {
        "key": task_id,
        "options": {"enabled": True, "limits": [1, 2, 3]},
    }
    capability_id = "fake.side_effect" if side_effecting else "fake.read"
    return Task(
        id=task_id,
        operation_id=OPERATION_ID,
        turn_id=TURN_ID,
        call_id=call_id,
        capability_id=capability_id,
        executor_id=f"{capability_id}.executor",
        status=status,
        attempt=attempt,
        arguments=arguments,
        execution_facts=TaskExecutionFacts(
            capability_fingerprint=("sha256:" + (fingerprint_character.lower() * 64)),
            arguments_hash=_sha256(arguments),
            access_mode=access_mode,
            risk=risk,
            side_effecting=side_effecting,
            idempotent=idempotent,
            replay_safe=replay_safe,
            idempotency_key=idempotency_key,
            validation_facts=validation_facts or ActionValidationFacts(),
        ),
        created_at=timestamp,
        updated_at=timestamp,
        manual_recovery_reason=manual_recovery_reason,
    )


def _snapshot() -> OperationSnapshot:
    call_ids = (
        "call-active",
        "call-parent",
        "call-manual",
        "call-other",
        "call-appended",
    )
    trigger = AgentTrigger(
        id=TRIGGER_ID,
        agent_id=AGENT_ID,
        kind=TriggerKind.USER,
        source_id="user-task-projection",
        session_id=SESSION_ID,
        payload={"message": "Exercise the durable task projection."},
        created_at=NOW,
    )
    operation = Operation(
        id=OPERATION_ID,
        agent_id=AGENT_ID,
        trigger_id=trigger.id,
        session_id=SESSION_ID,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=40),
    )
    request = ModelRequest(
        operation_id=OPERATION_ID,
        turn_id=TURN_ID,
        messages=(
            CanonicalMessage(
                agent_id=AGENT_ID,
                operation_id=OPERATION_ID,
                session_id=SESSION_ID,
                turn_id=TURN_ID,
                role=MessageRole.USER,
                content=(TextBlock("Exercise the durable task projection."),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="fake.action",
                description="Materialize one deterministic task.",
                input_schema={"type": "object"},
            ),
        ),
    )
    response = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tuple(
            ToolCall(
                id=call_id,
                name="fake.action",
                arguments={"key": call_id},
            )
            for call_id in call_ids
        ),
    )
    model_call = ModelCall(
        id=MODEL_CALL_ID,
        operation_id=OPERATION_ID,
        turn_id=TURN_ID,
        provider_id="mock:task-projection",
        request=request,
        response=response,
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW + timedelta(seconds=1),
    )
    turn = Turn(
        id=TURN_ID,
        operation_id=OPERATION_ID,
        number=1,
        model_request_id=model_call.id,
        model_response_id=model_call.id,
        created_at=NOW,
    )

    active = _task(
        "task-active",
        "call-active",
        fingerprint_character="a",
        status=TaskStatus.CLAIMED,
        attempt=2,
        timestamp=NOW + timedelta(seconds=10),
    )
    parent = _task(
        "task-parent",
        "call-parent",
        fingerprint_character="b",
        status=TaskStatus.PENDING,
        attempt=1,
    )
    manual = _task(
        "task-manual",
        "call-manual",
        fingerprint_character="c",
        status=TaskStatus.MANUAL_RECOVERY_REQUIRED,
        attempt=1,
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=False,
        replay_safe=False,
        validation_facts=ActionValidationFacts(
            schema_version=1,
            validation_passed=True,
            in_scope=True,
            destructive=False,
            sensitivity_class="confidential",
            source_id="source-sqlite",
            resource_ids=("resource-marker",),
            resource_revisions=(("resource-marker", "sha256:" + ("d" * 64)),),
            source_revision="sqlite:data-version:4",
            freshness_state="current",
            impact={"affected_rows": 1, "bounded": True},
        ),
        manual_recovery_reason="unknown_side_effect_outcome",
        timestamp=NOW + timedelta(seconds=30),
    )
    other = _task(
        "task-other",
        "call-other",
        fingerprint_character="e",
        status=TaskStatus.CLAIMED,
        attempt=1,
        timestamp=NOW + timedelta(seconds=6),
    )
    dependencies = (
        TaskDependency(
            operation_id=OPERATION_ID,
            task_id=manual.id,
            prerequisite_task_id=parent.id,
        ),
        TaskDependency(
            operation_id=OPERATION_ID,
            task_id=manual.id,
            prerequisite_task_id=active.id,
        ),
        TaskDependency(
            operation_id=OPERATION_ID,
            task_id=active.id,
            prerequisite_task_id=parent.id,
        ),
    )
    leases = (
        TaskLease(
            operation_id=OPERATION_ID,
            task_id=active.id,
            attempt=1,
            fencing_token=5,
            holder_id="worker-prior",
            acquired_at=NOW + timedelta(seconds=2),
            expires_at=NOW + timedelta(seconds=10),
            started_at=NOW + timedelta(seconds=3),
            renewed_at=NOW + timedelta(seconds=5),
            released_at=NOW + timedelta(seconds=10),
            release_reason="expired_reclaimed",
        ),
        TaskLease(
            operation_id=OPERATION_ID,
            task_id=other.id,
            attempt=1,
            fencing_token=3,
            holder_id="worker-other",
            acquired_at=NOW + timedelta(seconds=6),
            expires_at=NOW + timedelta(seconds=35),
        ),
        TaskLease(
            operation_id=OPERATION_ID,
            task_id=active.id,
            attempt=2,
            fencing_token=9,
            holder_id="worker-current",
            acquired_at=NOW + timedelta(seconds=10),
            expires_at=NOW + timedelta(seconds=40),
        ),
    )
    event = RuntimeEvent(
        id="event-operation-created",
        type="operation.created",
        agent_id=AGENT_ID,
        operation_id=OPERATION_ID,
        session_id=SESSION_ID,
        payload={"operation_id": OPERATION_ID},
        created_at=NOW,
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.AWAITING_EXECUTION,
            turn_count=1,
            action_count=4,
        ),
        budgets=LoopBudgets(),
        turns=(turn,),
        model_calls=(model_call,),
        readiness=(),
        tasks=(active, parent, manual, other),
        task_dependencies=dependencies,
        task_leases=leases,
        evidence=(),
        observations=(),
        events=(event,),
    )


def _appended_candidate(snapshot: OperationSnapshot) -> OperationSnapshot:
    appended = _task(
        "task-appended",
        "call-appended",
        fingerprint_character="d",
        status=TaskStatus.PENDING,
        attempt=1,
        timestamp=NOW + timedelta(seconds=50),
    )
    dependency = TaskDependency(
        operation_id=OPERATION_ID,
        task_id=appended.id,
        prerequisite_task_id="task-parent",
    )
    event = RuntimeEvent(
        id="event-task-appended",
        type="task.created",
        agent_id=AGENT_ID,
        operation_id=OPERATION_ID,
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        model_call_id=MODEL_CALL_ID,
        call_id=appended.call_id,
        task_id=appended.id,
        capability_id=appended.capability_id,
        executor_id=appended.executor_id,
        payload={"task_id": appended.id},
        created_at=NOW + timedelta(seconds=50),
    )
    return replace(
        snapshot,
        operation=replace(
            snapshot.operation,
            updated_at=NOW + timedelta(seconds=50),
        ),
        loop_state=replace(snapshot.loop_state, action_count=5),
        tasks=(*snapshot.tasks, appended),
        task_dependencies=(*snapshot.task_dependencies, dependency),
        events=(*snapshot.events, event),
    )


def _raw_rows(path: Path, table: str, columns: str) -> tuple[tuple[object, ...], ...]:
    connection = sqlite3.connect(path)
    try:
        return tuple(
            connection.execute(
                f"SELECT {columns} FROM {table} "
                "WHERE operation_id = ? ORDER BY position",
                (OPERATION_ID,),
            ).fetchall()
        )
    finally:
        connection.close()


async def _create_and_close(path: Path) -> OperationSnapshot:
    snapshot = _snapshot()
    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(snapshot)
    finally:
        await store.close()
    return snapshot


def _mutate(path: Path, statement: str, parameters: tuple[object, ...]) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(statement, parameters)
        connection.commit()
    finally:
        connection.close()


async def _assert_corrupt_load(path: Path) -> None:
    store = await SQLiteOperationStore.open(path)
    try:
        with pytest.raises(SQLiteCorruptionError):
            await store.load(OPERATION_ID)
    finally:
        await store.close()


async def test_task_projection_round_trips_exactly_across_reopen_and_global_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = await _create_and_close(path)

    reopened = await SQLiteOperationStore.open(path)
    try:
        loaded = await reopened.load(OPERATION_ID)
    finally:
        await reopened.close()

    assert loaded.revision == 1
    assert loaded.snapshot == snapshot
    assert loaded.snapshot.tasks[0].attempt == 2
    assert loaded.snapshot.tasks[0].execution_facts == (
        snapshot.tasks[0].execution_facts
    )
    assert loaded.snapshot.tasks[2].manual_recovery_reason == (
        "unknown_side_effect_outcome"
    )
    validation = loaded.snapshot.tasks[2].execution_facts.validation_facts
    assert validation.schema_version == 1
    assert validation.source_id == "source-sqlite"
    assert validation.source_ids == ("source-sqlite",)
    assert validation.source_revisions == (("source-sqlite", "sqlite:data-version:4"),)
    assert validation.freshness_state == "current"
    assert validation.resource_ids == ("resource-marker",)
    assert isinstance(validation.impact, FrozenJsonObject)
    assert validation.impact.to_dict() == {
        "affected_rows": 1,
        "bounded": True,
    }
    assert validation.fingerprint is not None
    assert _raw_rows(
        path,
        "task_dependencies",
        "task_id, prerequisite_task_id",
    ) == tuple(
        (dependency.task_id, dependency.prerequisite_task_id)
        for dependency in snapshot.task_dependencies
    )
    assert _raw_rows(
        path,
        "task_leases",
        "task_id, attempt, fencing_token",
    ) == tuple(
        (lease.task_id, lease.attempt, lease.fencing_token)
        for lease in snapshot.task_leases
    )


async def test_task_projection_round_trips_exact_multi_source_authority(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = _snapshot()
    validation = ActionValidationFacts(
        schema_version=1,
        validation_passed=True,
        in_scope=True,
        destructive=False,
        sensitivity_class="confidential",
        source_ids=("source-file", "source-sqlite"),
        source_revisions=(
            ("source-file", "sha256:" + ("1" * 64)),
            ("source-sqlite", "sqlite:data-version:4"),
        ),
        resource_ids=("resource-file", "resource-sqlite"),
        resource_revisions=(
            ("resource-file", "sha256:" + ("2" * 64)),
            ("resource-sqlite", "sha256:" + ("3" * 64)),
        ),
        freshness_state="current",
    )
    comparison = _task(
        "task-appended",
        "call-appended",
        fingerprint_character="d",
        status=TaskStatus.PENDING,
        attempt=1,
        validation_facts=validation,
        timestamp=NOW + timedelta(seconds=50),
    )
    candidate = replace(
        snapshot,
        operation=replace(
            snapshot.operation,
            updated_at=NOW + timedelta(seconds=50),
        ),
        loop_state=replace(snapshot.loop_state, action_count=5),
        tasks=(*snapshot.tasks, comparison),
    )

    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(candidate)
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        loaded = await reopened.load(OPERATION_ID)
    finally:
        await reopened.close()

    loaded_validation = loaded.snapshot.tasks[-1].execution_facts.validation_facts
    assert loaded_validation == validation
    assert loaded_validation.source_id is None
    assert _raw_rows(
        path,
        "tasks",
        "validation_source_id",
    )[
        -1
    ] == ("source-file",)


async def test_generic_task_and_dependency_suffixes_persist_when_still_pending(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = _snapshot()
    candidate = _appended_candidate(snapshot)
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)
        committed = await store.commit(
            candidate,
            expected_revision=created.operation.revision,
        )
        assert committed.operation.snapshot == candidate
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        loaded = await reopened.load(OPERATION_ID)
        assert loaded.revision == 2
        assert loaded.snapshot == candidate
    finally:
        await reopened.close()


async def test_generic_dependency_suffix_on_existing_pending_task_persists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = _snapshot()
    dependency = TaskDependency(
        operation_id=OPERATION_ID,
        task_id="task-parent",
        prerequisite_task_id="task-other",
    )
    event = RuntimeEvent(
        id="event-existing-task-dependency-added",
        type="task.dependency_added",
        agent_id=AGENT_ID,
        operation_id=OPERATION_ID,
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        model_call_id=MODEL_CALL_ID,
        call_id="call-parent",
        task_id="task-parent",
        payload={"prerequisite_task_id": "task-other"},
        created_at=NOW + timedelta(seconds=45),
    )
    candidate = replace(
        snapshot,
        operation=replace(
            snapshot.operation,
            updated_at=NOW + timedelta(seconds=45),
        ),
        task_dependencies=(*snapshot.task_dependencies, dependency),
        events=(*snapshot.events, event),
    )

    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)
        await store.commit(candidate, expected_revision=created.operation.revision)
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        loaded = await reopened.load(OPERATION_ID)
        assert loaded.snapshot == candidate
    finally:
        await reopened.close()


async def test_generic_existing_task_manual_recovery_fields_persist(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = _snapshot()
    recovered_parent = replace(
        snapshot.tasks[1],
        status=TaskStatus.MANUAL_RECOVERY_REQUIRED,
        updated_at=NOW + timedelta(seconds=45),
        manual_recovery_reason="operator_must_verify_external_outcome",
    )
    event = RuntimeEvent(
        id="event-existing-task-manual-recovery",
        type="task.manual_recovery_required",
        agent_id=AGENT_ID,
        operation_id=OPERATION_ID,
        session_id=SESSION_ID,
        turn_id=TURN_ID,
        model_call_id=MODEL_CALL_ID,
        call_id=recovered_parent.call_id,
        task_id=recovered_parent.id,
        capability_id=recovered_parent.capability_id,
        executor_id=recovered_parent.executor_id,
        payload={"reason": recovered_parent.manual_recovery_reason},
        created_at=NOW + timedelta(seconds=45),
    )
    candidate = replace(
        snapshot,
        operation=replace(
            snapshot.operation,
            updated_at=NOW + timedelta(seconds=45),
        ),
        tasks=(snapshot.tasks[0], recovered_parent, *snapshot.tasks[2:]),
        events=(*snapshot.events, event),
    )

    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)
        await store.commit(candidate, expected_revision=created.operation.revision)
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        loaded = await reopened.load(OPERATION_ID)
        assert loaded.snapshot == candidate
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    ("statement", "parameters", "message"),
    (
        (
            "UPDATE task_dependencies SET prerequisite_task_id = ? "
            "WHERE operation_id = ? AND position = 0",
            ("task-other", OPERATION_ID),
            "append-only",
        ),
        (
            "DELETE FROM task_dependencies " "WHERE operation_id = ? AND position = 0",
            (OPERATION_ID,),
            "append-only",
        ),
        (
            "UPDATE task_leases SET holder_id = ? "
            "WHERE operation_id = ? AND position = 2",
            ("different-holder", OPERATION_ID),
            "identity is immutable",
        ),
        (
            "DELETE FROM task_leases WHERE operation_id = ? AND position = 0",
            (OPERATION_ID,),
            "append-only",
        ),
        (
            "UPDATE task_leases SET release_reason = ? "
            "WHERE operation_id = ? AND position = 0",
            ("rewritten-release", OPERATION_ID),
            "released task lease is immutable",
        ),
    ),
    ids=(
        "dependency-update",
        "dependency-delete",
        "lease-identity-update",
        "lease-delete",
        "released-lease-update",
    ),
)
async def test_task_projection_schema_rejects_history_rewrites(
    tmp_path: Path,
    statement: str,
    parameters: tuple[object, ...],
    message: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)

    with pytest.raises(sqlite3.IntegrityError, match=message):
        _mutate(path, statement, parameters)


@pytest.mark.parametrize(
    ("attempt", "fencing_token"),
    ((2, 10), (3, 9)),
    ids=("duplicate-attempt", "duplicate-fence"),
)
async def test_task_lease_schema_rejects_duplicate_attempt_or_fence(
    tmp_path: Path,
    attempt: int,
    fencing_token: int,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)

    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        _mutate(
            path,
            "INSERT INTO task_leases("
            "operation_id, position, task_id, attempt, fencing_token, holder_id, "
            "acquired_at, expires_at, released_at, release_reason"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                OPERATION_ID,
                3,
                "task-active",
                attempt,
                fencing_token,
                "worker-duplicate-identity",
                "2026-07-17T19:00:41.000000Z",
                "2026-07-17T19:00:50.000000Z",
                "2026-07-17T19:00:50.000000Z",
                "released-test-row",
            ),
        )


@pytest.mark.parametrize("table", ("task_dependencies", "task_leases"))
async def test_task_projection_composite_foreign_keys_reject_cross_operation_rows(
    tmp_path: Path,
    table: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    encoded_now = "2026-07-17T19:01:00.000000Z"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO triggers("
            "id, agent_id, kind, source_id, payload_json, created_at, session_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "trigger-other-operation",
                "agent-other-operation",
                "internal",
                "source-other-operation",
                "{}",
                encoded_now,
                None,
            ),
        )
        connection.execute(
            "INSERT INTO operations("
            "id, revision, agent_id, trigger_id, status, created_at, updated_at, "
            "session_id, final_text, terminal_reason"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "operation-other",
                1,
                "agent-other-operation",
                "trigger-other-operation",
                "running",
                encoded_now,
                encoded_now,
                None,
                None,
                None,
            ),
        )
        if table == "task_dependencies":
            statement = (
                "INSERT INTO task_dependencies("
                "operation_id, position, task_id, prerequisite_task_id"
                ") VALUES (?, ?, ?, ?)"
            )
            parameters: tuple[object, ...] = (
                "operation-other",
                0,
                "task-active",
                "task-parent",
            )
        else:
            statement = (
                "INSERT INTO task_leases("
                "operation_id, position, task_id, attempt, fencing_token, "
                "holder_id, acquired_at, expires_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)"
            )
            parameters = (
                "operation-other",
                0,
                "task-active",
                1,
                1,
                "worker-cross-operation",
                encoded_now,
                "2026-07-17T19:02:00.000000Z",
            )

        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
            connection.execute(statement, parameters)
    finally:
        connection.rollback()
        connection.close()


async def test_task_lease_schema_allows_lifecycle_checkpoint_updates(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    started_at = "2026-07-17T19:00:11.000000Z"
    renewed_at = "2026-07-17T19:00:20.000000Z"
    expires_at = "2026-07-17T19:00:50.000000Z"
    _mutate(
        path,
        "UPDATE task_leases SET started_at = ?, renewed_at = ?, expires_at = ? "
        "WHERE operation_id = ? AND task_id = ? AND attempt = 2",
        (started_at, renewed_at, expires_at, OPERATION_ID, "task-active"),
    )

    store = await SQLiteOperationStore.open(path)
    try:
        loaded = await store.load(OPERATION_ID)
    finally:
        await store.close()
    updated = loaded.snapshot.task_leases[2]
    assert updated.started_at == NOW + timedelta(seconds=11)
    assert updated.renewed_at == NOW + timedelta(seconds=20)
    assert updated.expires_at == NOW + timedelta(seconds=50)


async def test_task_dependency_position_gap_is_typed_corruption(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        "INSERT INTO task_dependencies("
        "operation_id, position, task_id, prerequisite_task_id"
        ") VALUES (?, ?, ?, ?)",
        (OPERATION_ID, 4, "task-parent", "task-other"),
    )

    await _assert_corrupt_load(path)


async def test_task_dependency_cycle_is_typed_corruption(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        "INSERT INTO task_dependencies("
        "operation_id, position, task_id, prerequisite_task_id"
        ") VALUES (?, ?, ?, ?)",
        (OPERATION_ID, 3, "task-parent", "task-manual"),
    )

    await _assert_corrupt_load(path)


async def test_task_lease_overlap_is_typed_corruption(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        "UPDATE task_leases SET released_at = ?, release_reason = ? "
        "WHERE operation_id = ? AND task_id = ? AND attempt = 2",
        (
            "2026-07-17T19:00:20.000000Z",
            "completed-before-reclaim",
            OPERATION_ID,
            "task-active",
        ),
    )
    _mutate(
        path,
        "INSERT INTO task_leases("
        "operation_id, position, task_id, attempt, fencing_token, holder_id, "
        "acquired_at, expires_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            OPERATION_ID,
            3,
            "task-active",
            3,
            10,
            "worker-overlap",
            "2026-07-17T19:00:19.000000Z",
            "2026-07-17T19:00:50.000000Z",
        ),
    )

    await _assert_corrupt_load(path)


@pytest.mark.parametrize(
    ("column", "corrupt_value"),
    (
        ("status", "executing"),
        ("access_mode", "append"),
        ("risk", "critical"),
    ),
)
async def test_task_projection_rejects_corrupt_enums(
    tmp_path: Path,
    column: str,
    corrupt_value: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        f"UPDATE tasks SET {column} = ? WHERE id = ?",
        (corrupt_value, "task-active"),
    )

    await _assert_corrupt_load(path)


async def test_task_projection_rejects_unknown_validation_schema_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        "UPDATE tasks SET validation_schema_version = 2 WHERE id = ?",
        ("task-manual",),
    )

    await _assert_corrupt_load(path)


@pytest.mark.parametrize(
    ("column", "corrupt_value"),
    (
        ("validation_source_ids_json", '["source-other"]'),
        ("validation_resource_ids_json", '["resource-other"]'),
        (
            "validation_source_revisions_json",
            '[["source-sqlite","sqlite:data-version:other"]]',
        ),
    ),
)
async def test_task_projection_rejects_tampered_read_authority(
    tmp_path: Path,
    column: str,
    corrupt_value: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        f"UPDATE tasks SET {column} = ? WHERE id = ?",
        (corrupt_value, "task-manual"),
    )

    await _assert_corrupt_load(path)


@pytest.mark.parametrize("column", ("side_effecting", "idempotent", "replay_safe"))
async def test_task_projection_rejects_noncanonical_booleans(
    tmp_path: Path,
    column: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        f"UPDATE tasks SET {column} = 2 WHERE id = ?",
        ("task-active",),
    )

    await _assert_corrupt_load(path)


@pytest.mark.parametrize(
    "column",
    ("capability_fingerprint", "arguments_hash"),
)
async def test_task_projection_rejects_noncanonical_hashes(
    tmp_path: Path,
    column: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        f"UPDATE tasks SET {column} = ? WHERE id = ?",
        ("sha256:" + ("A" * 64), "task-active"),
    )

    await _assert_corrupt_load(path)


@pytest.mark.parametrize(
    "column",
    ("expires_at", "started_at", "renewed_at", "released_at"),
)
async def test_task_lease_projection_rejects_noncanonical_times(
    tmp_path: Path,
    column: str,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        f"UPDATE task_leases SET {column} = ? "
        "WHERE operation_id = ? AND task_id = ? AND attempt = 2",
        ("2026-07-17T19:00:00+00:00", OPERATION_ID, "task-active"),
    )

    await _assert_corrupt_load(path)


async def test_task_lease_projection_rejects_acquisition_identity_mutation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)

    with pytest.raises(sqlite3.IntegrityError, match="identity is immutable"):
        _mutate(
            path,
            "UPDATE task_leases SET acquired_at = ? "
            "WHERE operation_id = ? AND task_id = ? AND attempt = 2",
            ("2026-07-17T19:00:00+00:00", OPERATION_ID, "task-active"),
        )


@pytest.mark.parametrize(
    ("statement", "parameters"),
    (
        (
            "UPDATE task_leases SET released_at = ? "
            "WHERE operation_id = ? AND task_id = ? AND attempt = 2",
            (
                "2026-07-17T19:00:20.000000Z",
                OPERATION_ID,
                "task-active",
            ),
        ),
        (
            "UPDATE task_leases SET release_reason = ? "
            "WHERE operation_id = ? AND task_id = ? AND attempt = 2",
            ("released_without_time", OPERATION_ID, "task-active"),
        ),
    ),
    ids=("released_without_reason", "reason_without_release"),
)
async def test_task_lease_projection_rejects_unpaired_release_fields(
    tmp_path: Path,
    statement: str,
    parameters: tuple[object, ...],
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(path, statement, parameters)

    await _assert_corrupt_load(path)


async def test_task_lease_projection_rejects_duplicate_active_attempts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE constraint failed"):
        _mutate(
            path,
            "INSERT INTO task_leases("
            "operation_id, position, task_id, attempt, fencing_token, holder_id, "
            "acquired_at, expires_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                OPERATION_ID,
                3,
                "task-active",
                3,
                10,
                "worker-duplicate-active",
                "2026-07-17T19:00:41.000000Z",
                "2026-07-17T19:01:00.000000Z",
            ),
        )


async def test_task_projection_rejects_manual_recovery_without_reason(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    await _create_and_close(path)
    _mutate(
        path,
        "UPDATE tasks SET manual_recovery_reason = NULL WHERE id = ?",
        ("task-manual",),
    )

    await _assert_corrupt_load(path)
