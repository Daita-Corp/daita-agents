from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from daita.events.models import CommittedEvent, EventCursor, RuntimeEvent
from daita.events.protocols import EventCursorMismatchError
from daita.loop.models import LoopBudgets, LoopState
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
_DELIVERY_TIMEOUT_SECONDS = 0.75


def _snapshot(
    *,
    agent_id: str,
    operation_id: str,
    event_types: tuple[str, ...] = ("operation.created",),
) -> OperationSnapshot:
    events = tuple(
        RuntimeEvent(
            id=f"{operation_id}:event:{position}:{event_type}",
            type=event_type,
            agent_id=agent_id,
            operation_id=operation_id,
            created_at=NOW + timedelta(microseconds=position),
            payload={"position": position},
        )
        for position, event_type in enumerate(event_types)
    )
    trigger_id = f"{operation_id}:trigger"
    return OperationSnapshot(
        trigger=AgentTrigger(
            id=trigger_id,
            agent_id=agent_id,
            kind=TriggerKind.INTERNAL,
            source_id=f"{operation_id}:source",
            payload={"operation_id": operation_id},
            created_at=NOW,
        ),
        operation=Operation(
            id=operation_id,
            agent_id=agent_id,
            trigger_id=trigger_id,
            status=OperationStatus.RUNNING,
            created_at=NOW,
            updated_at=max((event.created_at for event in events), default=NOW),
        ),
        loop_state=LoopState(),
        budgets=LoopBudgets(),
        turns=(),
        model_calls=(),
        readiness=(),
        tasks=(),
        evidence=(),
        observations=(),
        events=events,
    )


def _append_event(
    snapshot: OperationSnapshot,
    *,
    event_type: str,
    seconds: int,
) -> OperationSnapshot:
    event = RuntimeEvent(
        id=f"{snapshot.operation.id}:event:{len(snapshot.events)}:{event_type}",
        type=event_type,
        agent_id=snapshot.operation.agent_id,
        operation_id=snapshot.operation.id,
        created_at=NOW + timedelta(seconds=seconds),
        payload={"revision_event": len(snapshot.events)},
    )
    return replace(
        snapshot,
        operation=replace(snapshot.operation, updated_at=event.created_at),
        events=(*snapshot.events, event),
    )


def _install_abort_trigger(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("""
            CREATE TRIGGER contract_abort_subscription_event
            BEFORE INSERT ON runtime_events
            WHEN NEW.type = 'contract.force_abort'
            BEGIN
                SELECT RAISE(ABORT, 'forced subscription event failure');
            END
            """)
        connection.commit()
    finally:
        connection.close()


def _drop_abort_trigger(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER contract_abort_subscription_event")
        connection.commit()
    finally:
        connection.close()


async def _next(stream: AsyncGenerator[CommittedEvent, None]) -> CommittedEvent:
    return await anext(stream)


async def _take(stream: AsyncGenerator[CommittedEvent, None]) -> CommittedEvent:
    return await asyncio.wait_for(
        _next(stream),
        timeout=_DELIVERY_TIMEOUT_SECONDS,
    )


async def _stop(
    stream: AsyncGenerator[CommittedEvent, None],
    pending: asyncio.Task[CommittedEvent] | None = None,
) -> None:
    if pending is not None:
        if not pending.done():
            pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
    await stream.aclose()


async def test_subscribe_wakes_only_after_create_and_commit_are_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    initial = _snapshot(agent_id="agent-durable", operation_id="operation-durable")
    candidate = _append_event(
        initial,
        event_type="operation.checkpointed",
        seconds=1,
    )
    store = await SQLiteOperationStore.open(path)
    stream = store.subscribe("agent-durable", None)
    durable_rows_at_notification: list[tuple[str, ...]] = []
    original_notify = store._notify_committed_events

    def assert_durable_then_notify(events: tuple[RuntimeEvent, ...]) -> None:
        connection = sqlite3.connect(path)
        try:
            rows = tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT id FROM runtime_events "
                    "WHERE agent_id = ? ORDER BY agent_sequence",
                    ("agent-durable",),
                )
            )
        finally:
            connection.close()
        assert all(event.id in rows for event in events)
        durable_rows_at_notification.append(rows)
        original_notify(events)

    monkeypatch.setattr(store, "_notify_committed_events", assert_durable_then_notify)
    pending = asyncio.create_task(_next(stream))
    try:
        await asyncio.sleep(0)
        await store.create(initial)
        created = await asyncio.wait_for(
            pending,
            timeout=_DELIVERY_TIMEOUT_SECONDS,
        )
        pending = asyncio.create_task(_next(stream))
        await asyncio.sleep(0)
        await store.commit(candidate, expected_revision=1)
        committed = await asyncio.wait_for(
            pending,
            timeout=_DELIVERY_TIMEOUT_SECONDS,
        )

        assert created == CommittedEvent(
            EventCursor("agent-durable", 1),
            initial.events[0],
        )
        assert committed == CommittedEvent(
            EventCursor("agent-durable", 2),
            candidate.events[-1],
        )
        assert durable_rows_at_notification == [
            (initial.events[0].id,),
            (initial.events[0].id, candidate.events[-1].id),
        ]
    finally:
        await _stop(stream, pending)
        await store.close()


async def test_rolled_back_event_is_never_delivered_or_assigned_a_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sqlite_owner,
        "_COMMITTED_EVENT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    path = tmp_path / "state.db"
    rejected = _snapshot(
        agent_id="agent-rollback",
        operation_id="operation-rejected",
        event_types=("contract.force_abort",),
    )
    accepted = _snapshot(
        agent_id="agent-rollback",
        operation_id="operation-accepted",
    )
    store = await SQLiteOperationStore.open(path)
    _install_abort_trigger(path)
    stream = store.subscribe("agent-rollback", None)
    pending = asyncio.create_task(_next(stream))
    trigger_installed = True
    try:
        with pytest.raises(sqlite3.IntegrityError, match="subscription event"):
            await store.create(rejected)
        await asyncio.sleep(0.04)
        assert not pending.done()

        _drop_abort_trigger(path)
        trigger_installed = False
        await store.create(accepted)
        delivered = await asyncio.wait_for(
            pending,
            timeout=_DELIVERY_TIMEOUT_SECONDS,
        )
        assert delivered == CommittedEvent(
            EventCursor("agent-rollback", 1),
            accepted.events[0],
        )
        assert await store.read_after("agent-rollback", None, limit=10) == (delivered,)
    finally:
        if trigger_installed:
            _drop_abort_trigger(path)
        await _stop(stream, pending)
        await store.close()


async def test_failed_local_wake_is_nonfatal_and_polling_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sqlite_owner,
        "_COMMITTED_EVENT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    snapshot = _snapshot(agent_id="agent-wake", operation_id="operation-wake")
    store = await SQLiteOperationStore.open(tmp_path / "state.db")
    stream = store.subscribe("agent-wake", None)

    def failed_notification(events: tuple[RuntimeEvent, ...]) -> None:
        assert events == snapshot.events
        raise RuntimeError("local wake channel failed")

    monkeypatch.setattr(store, "_notify_committed_events", failed_notification)
    pending = asyncio.create_task(_next(stream))
    try:
        await asyncio.sleep(0)
        result = await store.create(snapshot)
        delivered = await asyncio.wait_for(
            pending,
            timeout=_DELIVERY_TIMEOUT_SECONDS,
        )
        assert result.committed_events == snapshot.events
        assert delivered == CommittedEvent(
            EventCursor("agent-wake", 1),
            snapshot.events[0],
        )
    finally:
        await _stop(stream, pending)
        await store.close()


async def test_commit_between_empty_read_and_wait_is_closed_by_double_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sqlite_owner,
        "_COMMITTED_EVENT_POLL_INTERVAL_SECONDS",
        10.0,
    )
    path = tmp_path / "state.db"
    snapshot = _snapshot(agent_id="agent-race", operation_id="operation-race")
    subscriber = await SQLiteOperationStore.open(path)
    writer = await SQLiteOperationStore.open(path)
    original_read_after = subscriber.read_after
    calls = 0

    async def commit_after_empty_read(
        agent_id: str,
        cursor: EventCursor | None,
        *,
        limit: int,
    ) -> tuple[CommittedEvent, ...]:
        nonlocal calls
        calls += 1
        page = await original_read_after(agent_id, cursor, limit=limit)
        if calls == 1:
            assert page == ()
            await writer.create(snapshot)
        return page

    monkeypatch.setattr(subscriber, "read_after", commit_after_empty_read)
    stream = subscriber.subscribe("agent-race", None)
    try:
        delivered = await _take(stream)
        assert delivered == CommittedEvent(
            EventCursor("agent-race", 1),
            snapshot.events[0],
        )
        assert calls >= 2
    finally:
        await _stop(stream)
        await asyncio.gather(subscriber.close(), writer.close())


async def test_reconnect_resumes_strictly_after_exact_cursor(tmp_path: Path) -> None:
    initial = _snapshot(
        agent_id="agent-resume",
        operation_id="operation-resume",
        event_types=("resume.one", "resume.two"),
    )
    third = _append_event(initial, event_type="resume.three", seconds=1)
    fourth = _append_event(third, event_type="resume.four", seconds=2)
    store = await SQLiteOperationStore.open(tmp_path / "state.db")
    try:
        await store.create(initial)
        first_stream = store.subscribe("agent-resume", None)
        first = await _take(first_stream)
        second = await _take(first_stream)
        await first_stream.aclose()

        await store.commit(third, expected_revision=1)
        resumed = store.subscribe("agent-resume", second.cursor)
        pending: asyncio.Task[CommittedEvent] | None = None
        try:
            replayed = await _take(resumed)
            pending = asyncio.create_task(_next(resumed))
            await asyncio.sleep(0)
            await store.commit(fourth, expected_revision=2)
            followed = await asyncio.wait_for(
                pending,
                timeout=_DELIVERY_TIMEOUT_SECONDS,
            )
        finally:
            await _stop(resumed, pending)

        assert (first.cursor.sequence, second.cursor.sequence) == (1, 2)
        assert replayed == CommittedEvent(
            EventCursor("agent-resume", 3),
            third.events[-1],
        )
        assert followed == CommittedEvent(
            EventCursor("agent-resume", 4),
            fourth.events[-1],
        )
    finally:
        await store.close()


async def test_cross_store_subscriber_observes_commits_by_polling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sqlite_owner,
        "_COMMITTED_EVENT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    path = tmp_path / "state.db"
    snapshot = _snapshot(agent_id="agent-cross", operation_id="operation-cross")
    subscriber = await SQLiteOperationStore.open(path)
    writer = await SQLiteOperationStore.open(path)
    stream = subscriber.subscribe("agent-cross", None)
    pending = asyncio.create_task(_next(stream))
    try:
        await asyncio.sleep(0.02)
        assert not pending.done()
        await writer.create(snapshot)
        delivered = await asyncio.wait_for(
            pending,
            timeout=_DELIVERY_TIMEOUT_SECONDS,
        )
        assert delivered == CommittedEvent(
            EventCursor("agent-cross", 1),
            snapshot.events[0],
        )
    finally:
        await _stop(stream, pending)
        await asyncio.gather(subscriber.close(), writer.close())


async def test_slow_subscriber_reads_bounded_pages_without_loss_or_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sqlite_owner,
        "_COMMITTED_EVENT_SUBSCRIPTION_BATCH_SIZE",
        2,
    )
    snapshot = _snapshot(
        agent_id="agent-slow",
        operation_id="operation-slow",
        event_types=tuple(f"slow.{index}" for index in range(7)),
    )
    store = await SQLiteOperationStore.open(tmp_path / "state.db")
    await store.create(snapshot)
    original_read_after = store.read_after
    reads: list[tuple[EventCursor | None, int]] = []

    async def bounded_read(
        agent_id: str,
        cursor: EventCursor | None,
        *,
        limit: int,
    ) -> tuple[CommittedEvent, ...]:
        reads.append((cursor, limit))
        return await original_read_after(agent_id, cursor, limit=limit)

    monkeypatch.setattr(store, "read_after", bounded_read)
    stream = store.subscribe("agent-slow", None)
    try:
        delivered: list[CommittedEvent] = []
        for _ in snapshot.events:
            delivered.append(await _take(stream))
            await asyncio.sleep(0)

        assert tuple(item.cursor.sequence for item in delivered) == tuple(range(1, 8))
        assert tuple(item.event for item in delivered) == snapshot.events
        assert reads == [
            (None, 2),
            (EventCursor("agent-slow", 2), 2),
            (EventCursor("agent-slow", 4), 2),
            (EventCursor("agent-slow", 6), 2),
        ]
    finally:
        await _stop(stream)
        await store.close()


async def test_multiple_subscribers_receive_the_same_cursor_without_interference(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(
        agent_id="agent-multiple",
        operation_id="operation-multiple",
    )
    store = await SQLiteOperationStore.open(tmp_path / "state.db")
    first_stream = store.subscribe("agent-multiple", None)
    second_stream = store.subscribe("agent-multiple", None)
    first_pending = asyncio.create_task(_next(first_stream))
    second_pending = asyncio.create_task(_next(second_stream))
    try:
        await asyncio.sleep(0)
        await store.create(snapshot)
        first, second = await asyncio.gather(first_pending, second_pending)

        expected = CommittedEvent(
            EventCursor("agent-multiple", 1),
            snapshot.events[0],
        )
        assert first == expected
        assert second == expected
    finally:
        await _stop(first_stream, first_pending)
        await _stop(second_stream, second_pending)
        await store.close()


async def test_cancelling_and_closing_subscription_leaves_no_background_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sqlite_owner,
        "_COMMITTED_EVENT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    store = await SQLiteOperationStore.open(tmp_path / "state.db")
    current = asyncio.current_task()
    baseline = {
        task for task in asyncio.all_tasks() if task is not current and not task.done()
    }
    stream = store.subscribe("agent-cancel", None)
    pending = asyncio.create_task(
        _next(stream),
        name="contract-event-subscription",
    )
    try:
        await asyncio.sleep(0.03)
        assert not pending.done()
        pending.cancel()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await stream.aclose()
        await asyncio.sleep(0)

        leaked = {
            task
            for task in asyncio.all_tasks()
            if task is not current and not task.done() and task not in baseline
        }
        assert leaked == set()
    finally:
        if not pending.done():
            pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
        await store.close()


async def test_subscription_is_agent_scoped_and_rejects_foreign_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sqlite_owner,
        "_COMMITTED_EVENT_POLL_INTERVAL_SECONDS",
        0.01,
    )
    path = tmp_path / "state.db"
    agent_a = _snapshot(agent_id="agent-a", operation_id="operation-a")
    agent_b = _snapshot(agent_id="agent-b", operation_id="operation-b")
    store = await SQLiteOperationStore.open(path)
    stream = store.subscribe("agent-a", None)
    pending = asyncio.create_task(_next(stream))
    try:
        await store.create(agent_b)
        await asyncio.sleep(0.04)
        assert not pending.done()

        await store.create(agent_a)
        delivered = await asyncio.wait_for(
            pending,
            timeout=_DELIVERY_TIMEOUT_SECONDS,
        )
        assert delivered == CommittedEvent(
            EventCursor("agent-a", 1),
            agent_a.events[0],
        )

        foreign_cursor = EventCursor("agent-b", 1)
        with pytest.raises(EventCursorMismatchError) as mismatch:
            store.subscribe("agent-a", foreign_cursor)
        assert mismatch.value.requested_agent_id == "agent-a"
        assert mismatch.value.cursor == foreign_cursor
    finally:
        await _stop(stream, pending)
        await store.close()
