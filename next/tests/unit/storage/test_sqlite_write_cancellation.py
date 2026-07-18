from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3
import threading
from typing import Any, Literal

import pytest

from daita.events.models import RuntimeEvent
from daita.loop.models import LoopBudgets, LoopPhase, LoopState
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.operations.store import CommitResult, VersionedOperation
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import SQLiteOperationStore, SQLiteStoreError

NOW = datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc)
LATER = NOW + timedelta(seconds=1)


def _event(event_id: str, event_type: str, *, created_at: datetime) -> RuntimeEvent:
    return RuntimeEvent(
        id=event_id,
        type=event_type,
        agent_id="agent-write-cancellation",
        operation_id="operation-write-cancellation",
        payload={"event_id": event_id},
        created_at=created_at,
    )


def _initial_snapshot() -> OperationSnapshot:
    trigger = AgentTrigger(
        id="trigger-write-cancellation",
        agent_id="agent-write-cancellation",
        kind=TriggerKind.USER,
        source_id="user-write-cancellation",
        payload={"prompt": "persist before returning"},
        created_at=NOW,
    )
    operation = Operation(
        id="operation-write-cancellation",
        agent_id=trigger.agent_id,
        trigger_id=trigger.id,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
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
        events=(
            _event(
                "event-write-cancellation-1",
                "operation.created",
                created_at=NOW,
            ),
        ),
    )


def _advanced_snapshot(initial: OperationSnapshot) -> OperationSnapshot:
    return replace(
        initial,
        operation=replace(initial.operation, updated_at=LATER),
        loop_state=replace(
            initial.loop_state,
            phase=LoopPhase.AWAITING_MODEL,
            turn_count=1,
        ),
        events=(
            *initial.events,
            _event(
                "event-write-cancellation-2",
                "context.built",
                created_at=LATER,
            ),
        ),
    )


async def _wait_until_set(event: threading.Event) -> None:
    """Wait for a worker-thread checkpoint without blocking the event loop."""

    assert await asyncio.to_thread(event.wait, 2.0)


async def _invoke_write(
    store: SQLiteOperationStore,
    write_kind: Literal["create", "commit"],
    snapshot: OperationSnapshot,
) -> CommitResult:
    if write_kind == "create":
        return await store.create(snapshot)
    return await store.commit(snapshot, expected_revision=1)


def test_later_event_prefix_is_not_proof_of_an_ambiguous_commit() -> None:
    initial = _initial_snapshot()
    candidate_snapshot = _advanced_snapshot(initial)
    candidate = VersionedOperation(snapshot=candidate_snapshot, revision=2)
    successor_time = LATER + timedelta(seconds=1)
    successor_snapshot = replace(
        candidate_snapshot,
        operation=replace(
            candidate_snapshot.operation,
            updated_at=successor_time,
        ),
        events=(
            *candidate_snapshot.events,
            _event(
                "event-write-cancellation-3",
                "checkpoint.updated",
                created_at=successor_time,
            ),
        ),
    )
    observed = VersionedOperation(snapshot=successor_snapshot, revision=3)

    assert sqlite_owner._observed_commit_proves_result(observed, candidate) is False


@pytest.mark.parametrize("write_kind", ("create", "commit"))
async def test_cancelled_write_waits_for_transaction_and_reopens_exact_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_kind: Literal["create", "commit"],
) -> None:
    path = tmp_path / "state.db"
    real_connect = sqlite3.connect
    transaction_gate_armed = threading.Event()
    transaction_blocked = threading.Event()
    release_transaction = threading.Event()
    transaction_finished = threading.Event()
    gate_timed_out = threading.Event()
    caller_returned = threading.Event()
    statements_after_return: list[str] = []
    traced_statements: list[str] = []

    class TrackingConnection(sqlite3.Connection):
        def execute(
            self,
            sql: str,
            parameters: Any = (),
            /,
        ) -> sqlite3.Cursor:
            cursor = super().execute(sql, parameters)
            if sql.strip().upper() in {"COMMIT", "ROLLBACK"}:
                transaction_finished.set()
            return cursor

    def tracking_connect(
        database: str | bytes | Path,
        timeout: float = 5.0,
        detect_types: int = 0,
        isolation_level: (
            Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None
        ) = "DEFERRED",
        check_same_thread: bool = True,
        factory: type[sqlite3.Connection] = TrackingConnection,
        cached_statements: int = 128,
        uri: bool = False,
    ) -> sqlite3.Connection:
        del factory
        connection = real_connect(
            database,
            timeout=timeout,
            detect_types=detect_types,
            isolation_level=isolation_level,
            check_same_thread=check_same_thread,
            factory=TrackingConnection,
            cached_statements=cached_statements,
            uri=uri,
        )

        def trace(statement: str) -> None:
            traced_statements.append(statement)
            if caller_returned.is_set():
                statements_after_return.append(statement)
            # BEGIN has completed by the first traced statement for which the
            # connection reports an active transaction. Blocking here proves
            # the public await cannot outrun a worker that can still commit.
            if transaction_gate_armed.is_set() and connection.in_transaction:
                transaction_gate_armed.clear()
                transaction_blocked.set()
                if not release_transaction.wait(2.0):
                    gate_timed_out.set()

        connection.set_trace_callback(trace)
        return connection

    monkeypatch.setattr(sqlite_owner.sqlite3, "connect", tracking_connect)
    store = await SQLiteOperationStore.open(path)
    initial = _initial_snapshot()
    target = initial
    expected_revision = 1
    if write_kind == "commit":
        await store.create(initial)
        target = _advanced_snapshot(initial)
        expected_revision = 2

    transaction_finished.clear()
    transaction_gate_armed.set()
    writing = asyncio.create_task(_invoke_write(store, write_kind, target))
    returned_before_release = False
    try:
        await _wait_until_set(transaction_blocked)
        writing.cancel()
        await asyncio.sleep(0)
        returned_before_release = writing.done()
        if returned_before_release:
            caller_returned.set()

        release_transaction.set()
        with pytest.raises(asyncio.CancelledError):
            await writing
        caller_returned.set()
        await _wait_until_set(transaction_finished)
    finally:
        release_transaction.set()
        if not writing.done():
            writing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await writing

    statements_at_return = tuple(traced_statements)
    await asyncio.to_thread(lambda: None)

    assert returned_before_release is False
    assert gate_timed_out.is_set() is False
    assert statements_after_return == []
    assert tuple(traced_statements) == statements_at_return

    await store.close()
    monkeypatch.setattr(sqlite_owner.sqlite3, "connect", real_connect)

    reopened = await SQLiteOperationStore.open(path)
    try:
        loaded = await reopened.load(target.operation.id)
        assert loaded.revision == expected_revision
        assert loaded.snapshot == target
    finally:
        await reopened.close()


@pytest.mark.parametrize("write_kind", ("create", "commit"))
async def test_error_immediately_after_commit_is_reconciled_or_typed_unknown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    write_kind: Literal["create", "commit"],
) -> None:
    path = tmp_path / "state.db"
    real_connect = sqlite3.connect
    inject_post_commit_error = threading.Event()

    class PostCommitErrorConnection(sqlite3.Connection):
        def execute(
            self,
            sql: str,
            parameters: Any = (),
            /,
        ) -> sqlite3.Cursor:
            cursor = super().execute(sql, parameters)
            if inject_post_commit_error.is_set() and sql.strip().upper() == "COMMIT":
                inject_post_commit_error.clear()
                # SQLite has durably committed, but the adapter observes an
                # exception exactly where a real I/O boundary can be ambiguous.
                raise sqlite3.OperationalError("injected error after COMMIT")
            return cursor

    def post_commit_error_connect(
        database: str | bytes | Path,
        timeout: float = 5.0,
        detect_types: int = 0,
        isolation_level: (
            Literal["DEFERRED", "EXCLUSIVE", "IMMEDIATE"] | None
        ) = "DEFERRED",
        check_same_thread: bool = True,
        factory: type[sqlite3.Connection] = PostCommitErrorConnection,
        cached_statements: int = 128,
        uri: bool = False,
    ) -> sqlite3.Connection:
        del factory
        return real_connect(
            database,
            timeout=timeout,
            detect_types=detect_types,
            isolation_level=isolation_level,
            check_same_thread=check_same_thread,
            factory=PostCommitErrorConnection,
            cached_statements=cached_statements,
            uri=uri,
        )

    monkeypatch.setattr(
        sqlite_owner.sqlite3,
        "connect",
        post_commit_error_connect,
    )
    store = await SQLiteOperationStore.open(path)
    initial = _initial_snapshot()
    target = initial
    expected_revision = 1
    if write_kind == "commit":
        await store.create(initial)
        target = _advanced_snapshot(initial)
        expected_revision = 2

    inject_post_commit_error.set()
    outcome: CommitResult | BaseException
    try:
        outcome = await _invoke_write(store, write_kind, target)
    except BaseException as error:
        outcome = error

    if isinstance(outcome, BaseException):
        # If the adapter cannot prove the exact outcome by reading it back, it
        # must preserve ambiguity explicitly. A generic sqlite error falsely
        # tells callers that retrying the write is harmless.
        assert isinstance(outcome, SQLiteStoreError)
        assert type(outcome).__name__ == "SQLiteCommitOutcomeUnknownError"
        assert not isinstance(outcome, sqlite3.Error)
    else:
        assert outcome.operation.revision == expected_revision
        assert outcome.operation.snapshot == target
        assert outcome.committed_events == (
            target.events if write_kind == "create" else target.events[-1:]
        )

    await store.close()
    monkeypatch.setattr(sqlite_owner.sqlite3, "connect", real_connect)

    reopened = await SQLiteOperationStore.open(path)
    try:
        loaded = await reopened.load(target.operation.id)
        assert loaded.revision == expected_revision
        assert loaded.snapshot == target
    finally:
        await reopened.close()
