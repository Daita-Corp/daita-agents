from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
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
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
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
from daita.storage.sqlite import SQLiteCorruptionError, SQLiteOperationStore

OFFSET = timezone(timedelta(hours=5, minutes=30))
EARLY = datetime(2026, 7, 17, 8, 9, 10, 111_222, tzinfo=OFFSET)
MIDDLE = EARLY + timedelta(seconds=3)
LATE = EARLY + timedelta(seconds=7)

NORMALIZED_LIFECYCLE_TABLES = {
    "triggers",
    "operations",
    "loop_state",
    "turns",
    "model_calls",
    "readiness",
    "tasks",
    "task_dependencies",
    "task_leases",
    "task_evidence",
    "evidence",
    "observations",
    "runtime_events",
}


def _rich_json(label: str) -> dict[str, object]:
    return {
        "array": [None, True, 7, 1.25, label, {"nested": ["\u03b1", "\u03b2"]}],
        "empty": {},
        "label": label,
    }


def _maximal_snapshot() -> OperationSnapshot:
    operation_id = "operation-roundtrip"
    agent_id = "agent-roundtrip"
    session_id = "session-roundtrip"

    trigger = AgentTrigger(
        id="trigger-roundtrip",
        agent_id=agent_id,
        kind=TriggerKind.EVENT,
        source_id="source-roundtrip",
        session_id=session_id,
        payload=_rich_json("trigger"),
        created_at=EARLY,
    )
    operation = Operation(
        id=operation_id,
        agent_id=agent_id,
        trigger_id=trigger.id,
        session_id=session_id,
        status=OperationStatus.SUCCEEDED,
        final_text="Durable answer \u2713",
        terminal_reason="readiness_satisfied",
        created_at=EARLY,
        updated_at=LATE,
    )

    # Collection order is deliberately neither lexical, numeric, nor chronological.
    turn_z = Turn(
        id="turn-z",
        operation_id=operation_id,
        number=2,
        model_request_id="model-call-z",
        model_response_id="model-call-z",
        created_at=MIDDLE,
    )
    turn_a = Turn(
        id="turn-a",
        operation_id=operation_id,
        number=1,
        model_request_id="model-call-a",
        created_at=EARLY,
    )

    request_z = ModelRequest(
        operation_id=operation_id,
        turn_id=turn_z.id,
        messages=(
            CanonicalMessage(
                agent_id=agent_id,
                operation_id=operation_id,
                turn_id=turn_z.id,
                session_id=session_id,
                role=MessageRole.SYSTEM,
                content=(TextBlock("Use durable evidence."),),
            ),
            CanonicalMessage(
                agent_id=agent_id,
                operation_id=operation_id,
                turn_id=turn_z.id,
                session_id=session_id,
                role=MessageRole.USER,
                content=(TextBlock("Read both records."),),
            ),
            CanonicalMessage(
                agent_id=agent_id,
                operation_id=operation_id,
                turn_id=turn_z.id,
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=(TextBlock("Checking prior evidence."),),
                tool_calls=(
                    ToolCall(
                        id="prior-call",
                        name="fake.prior",
                        arguments=_rich_json("prior-call"),
                    ),
                ),
            ),
            CanonicalMessage(
                agent_id=agent_id,
                operation_id=operation_id,
                turn_id=turn_z.id,
                session_id=session_id,
                role=MessageRole.TOOL,
                content=(
                    ToolResultBlock(
                        call_id="prior-call",
                        output=_rich_json("prior-result"),
                        is_error=True,
                    ),
                ),
            ),
        ),
        tools=(
            ToolDefinition(
                name="fake.read",
                description="Read one deterministic record.",
                input_schema={
                    "type": "object",
                    "properties": {"key": {"type": "string"}},
                    "required": ["key"],
                    "additionalProperties": False,
                },
            ),
            ToolDefinition(
                name="fake.prior",
                description="Inspect a prior deterministic record.",
                input_schema={"type": "object"},
            ),
        ),
    )
    response_z = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        text="I will read both records.",
        tool_calls=(
            ToolCall(
                id="call-z",
                name="fake.read",
                arguments={"key": "z", "options": _rich_json("call-z")},
            ),
            ToolCall(
                id="call-a",
                name="fake.read",
                arguments={"key": "a", "options": _rich_json("call-a")},
            ),
        ),
        usage=ModelUsage(
            input_tokens=101,
            output_tokens=23,
            reasoning_tokens=5,
            cache_read_tokens=11,
            cache_write_tokens=13,
            estimated_cost_usd=Decimal("0.0012300"),
        ),
        provider_response_id="provider-response-z",
        provider_metadata=_rich_json("provider-response"),
    )
    model_call_z = ModelCall(
        id="model-call-z",
        operation_id=operation_id,
        turn_id=turn_z.id,
        provider_id="mock:roundtrip",
        request=request_z,
        response=response_z,
        status=ModelCallStatus.COMPLETED,
        cancellation_requested=True,
        created_at=MIDDLE,
        updated_at=LATE,
    )

    request_a = ModelRequest(
        operation_id=operation_id,
        turn_id=turn_a.id,
        messages=(
            CanonicalMessage(
                agent_id=agent_id,
                operation_id=operation_id,
                turn_id=turn_a.id,
                session_id=session_id,
                role=MessageRole.USER,
                content=(TextBlock("This provider call fails deterministically."),),
            ),
        ),
    )
    model_call_a = ModelCall(
        id="model-call-a",
        operation_id=operation_id,
        turn_id=turn_a.id,
        provider_id="mock:failing",
        request=request_a,
        status=ModelCallStatus.FAILED,
        error_code="provider_unavailable",
        cancellation_requested=True,
        created_at=EARLY,
        updated_at=MIDDLE,
    )

    evidence_z = Evidence(
        id="evidence-z",
        operation_id=operation_id,
        task_id="task-z",
        turn_id=turn_z.id,
        capability_id="fake.read",
        executor_id="fake.executor",
        kind="fake.record",
        schema_version=2,
        attempt=3,
        accepted=True,
        payload=_rich_json("evidence-z"),
        content_hash="sha256:" + "f" * 64,
        created_at=LATE,
    )
    evidence_a = Evidence(
        id="evidence-a",
        operation_id=operation_id,
        task_id="task-z",
        turn_id=turn_z.id,
        capability_id="fake.read",
        executor_id="fake.executor",
        kind="fake.record",
        schema_version=2,
        attempt=3,
        accepted=True,
        payload=_rich_json("evidence-a"),
        content_hash="sha256:" + "a" * 64,
        created_at=MIDDLE,
    )
    task_z = Task(
        id="task-z",
        operation_id=operation_id,
        turn_id=turn_z.id,
        call_id="call-z",
        capability_id="fake.read",
        executor_id="fake.executor",
        status=TaskStatus.SUCCEEDED,
        attempt=3,
        arguments={"key": "z", "options": _rich_json("task-z")},
        # This task-local order intentionally opposes aggregate evidence order.
        evidence_ids=(evidence_a.id, evidence_z.id),
        cancellation_requested=True,
        created_at=MIDDLE,
        updated_at=LATE,
    )
    task_a = Task(
        id="task-a",
        operation_id=operation_id,
        turn_id=turn_z.id,
        call_id="call-a",
        capability_id="fake.read",
        executor_id="fake.executor",
        status=TaskStatus.FAILED,
        attempt=2,
        arguments={"key": "a", "options": _rich_json("task-a")},
        error_code="executor_failed",
        cancellation_requested=True,
        created_at=EARLY,
        updated_at=MIDDLE,
    )

    readiness_late = Readiness(
        allowed=False,
        code="missing_nested_fact",
        message="A nested fact is still missing.",
        missing_facts=("alpha.total", "beta.status"),
        evaluated_at=LATE,
    )
    readiness_early = Readiness(
        allowed=True,
        code="ready",
        message="The answer is supported.",
        evaluated_at=EARLY,
    )

    observation_late = Observation(
        operation_id=operation_id,
        turn_id=turn_z.id,
        call_id=task_z.call_id,
        task_id=task_z.id,
        evidence_id=evidence_z.id,
        code="fake.read.succeeded",
        message="The later record was accepted.",
        payload=_rich_json("observation-z"),
        success=True,
        truncated=True,
        created_at=LATE,
    )
    observation_early = Observation(
        operation_id=operation_id,
        turn_id=turn_z.id,
        call_id=task_a.call_id,
        task_id=task_a.id,
        code="fake.read.failed",
        message="The earlier record failed.",
        payload=_rich_json("observation-a"),
        success=False,
        created_at=EARLY,
    )

    event_z = RuntimeEvent(
        id="event-z",
        type="evidence.accepted",
        agent_id=agent_id,
        operation_id=operation_id,
        session_id=session_id,
        turn_id=turn_z.id,
        model_call_id=model_call_z.id,
        call_id=task_z.call_id,
        task_id=task_z.id,
        evidence_id=evidence_z.id,
        capability_id=task_z.capability_id,
        executor_id=task_z.executor_id,
        payload=_rich_json("event-z"),
        created_at=LATE,
    )
    event_a = RuntimeEvent(
        id="event-a",
        type="turn.created",
        agent_id=agent_id,
        operation_id=operation_id,
        session_id=session_id,
        turn_id=turn_a.id,
        payload=_rich_json("event-a"),
        created_at=EARLY,
    )
    event_m = RuntimeEvent(
        id="event-m",
        type="task.failed",
        agent_id=agent_id,
        operation_id=operation_id,
        session_id=session_id,
        turn_id=turn_z.id,
        model_call_id=model_call_z.id,
        call_id=task_a.call_id,
        task_id=task_a.id,
        capability_id=task_a.capability_id,
        executor_id=task_a.executor_id,
        payload=_rich_json("event-m"),
        created_at=MIDDLE,
    )

    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.TERMINAL,
            turn_count=9,
            action_count=8,
            repair_count=7,
            identical_failure_count=6,
            observation_characters=5_432,
            input_tokens=321,
            output_tokens=123,
            estimated_cost_usd=Decimal("12.3400"),
            interruption_reason="host_restart",
            final_answer_candidate="Durable answer \u2713",
            no_progress_fingerprints=("fingerprint-z", "fingerprint-a"),
        ),
        budgets=LoopBudgets(
            max_turns=19,
            max_actions=18,
            max_repairs=17,
            max_identical_failures=16,
            max_observation_characters=150_001,
            max_total_tokens=250_002,
            max_wall_time_seconds=301.25,
            task_timeout_seconds=31.75,
            max_estimated_cost_usd=Decimal("98.7600"),
        ),
        turns=(turn_z, turn_a),
        model_calls=(model_call_z, model_call_a),
        readiness=(readiness_late, readiness_early, readiness_late),
        tasks=(task_z, task_a),
        evidence=(evidence_z, evidence_a),
        observations=(observation_late, observation_early, observation_late),
        events=(event_z, event_a, event_m),
    )


def _table_columns(connection: sqlite3.Connection, table: str) -> set[str]:
    return {
        str(row[1])
        for row in connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    }


async def test_maximal_snapshot_round_trips_through_lookups_and_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = _maximal_snapshot()

    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)

        assert created.operation.revision == 1
        assert created.operation.snapshot == snapshot
        assert created.committed_events == snapshot.events
        assert await store.load(snapshot.operation.id) == created.operation
        assert await store.load_by_trigger(snapshot.trigger.id) == created.operation
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        assert await reopened.load(snapshot.operation.id) == created.operation
        assert await reopened.load_by_trigger(snapshot.trigger.id) == created.operation
    finally:
        await reopened.close()


async def test_evidence_blob_id_round_trips_with_nullable_legacy_rows(
    tmp_path: Path,
) -> None:
    path = tmp_path / "evidence-blobs.db"
    original = _maximal_snapshot()
    linked = replace(original.evidence[0], blob_id="blob-evidence-z")
    unlinked = replace(original.evidence[1], blob_id=None)
    snapshot = replace(original, evidence=(linked, unlinked))

    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(snapshot)
        assert created.operation.snapshot == snapshot
        assert await store.load(snapshot.operation.id) == created.operation
    finally:
        await store.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT id, blob_id FROM evidence "
            "WHERE operation_id = ? ORDER BY position",
            (snapshot.operation.id,),
        ).fetchall() == [
            (linked.id, linked.blob_id),
            (unlinked.id, None),
        ]
    finally:
        connection.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        assert await reopened.load(snapshot.operation.id) == created.operation
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    "corrupt_blob_id",
    (
        "   ",
        sqlite3.Binary(b"blob-id-must-be-text"),
    ),
    ids=("blank-text", "non-text"),
)
async def test_evidence_blob_id_codec_rejects_malformed_projection(
    tmp_path: Path,
    corrupt_blob_id: object,
) -> None:
    path = tmp_path / "corrupt-evidence-blob.db"
    snapshot = _maximal_snapshot()
    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(snapshot)
    finally:
        await store.close()

    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE evidence SET blob_id = ? WHERE operation_id = ? AND id = ?",
            (
                corrupt_blob_id,
                snapshot.operation.id,
                snapshot.evidence[0].id,
            ),
        )
        connection.commit()
    finally:
        connection.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        with pytest.raises(SQLiteCorruptionError):
            await reopened.load(snapshot.operation.id)
    finally:
        await reopened.close()


async def test_raw_schema_normalizes_rows_and_preserves_independent_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshot = _maximal_snapshot()
    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(snapshot)
    finally:
        await store.close()

    connection = sqlite3.connect(path)
    try:
        table_sql = {
            str(name): str(sql or "")
            for name, sql in connection.execute(
                "SELECT name, sql FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        assert NORMALIZED_LIFECYCLE_TABLES <= table_sql.keys()
        assert all(
            "snapshot_json" not in _table_columns(connection, table)
            for table in table_sql
        )
        assert all("snapshot_json" not in sql.lower() for sql in table_sql.values())

        assert connection.execute(
            "SELECT revision FROM operations WHERE id = ?",
            (snapshot.operation.id,),
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT id FROM turns WHERE operation_id = ? ORDER BY position",
            (snapshot.operation.id,),
        ).fetchall() == [(turn.id,) for turn in snapshot.turns]
        assert connection.execute(
            "SELECT id FROM evidence WHERE operation_id = ? ORDER BY position",
            (snapshot.operation.id,),
        ).fetchall() == [(item.id,) for item in snapshot.evidence]
        assert connection.execute(
            "SELECT evidence_id FROM task_evidence "
            "WHERE operation_id = ? AND task_id = ? ORDER BY position",
            (snapshot.operation.id, "task-z"),
        ).fetchall() == [(item,) for item in snapshot.tasks[0].evidence_ids]
        assert connection.execute(
            "SELECT position FROM readiness "
            "WHERE operation_id = ? ORDER BY position",
            (snapshot.operation.id,),
        ).fetchall() == [(0,), (1,), (2,)]
        assert connection.execute(
            "SELECT position FROM observations "
            "WHERE operation_id = ? ORDER BY position",
            (snapshot.operation.id,),
        ).fetchall() == [(0,), (1,), (2,)]
        assert connection.execute(
            "SELECT id FROM runtime_events " "WHERE operation_id = ? ORDER BY position",
            (snapshot.operation.id,),
        ).fetchall() == [(event.id,) for event in snapshot.events]
    finally:
        connection.close()
