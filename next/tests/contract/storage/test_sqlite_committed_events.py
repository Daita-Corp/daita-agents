from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sqlite3

import pytest

from daita.events.models import CommittedEvent, EventCursor, RuntimeEvent
from daita.events.protocols import (
    EventCursorMismatchError,
    EventCursorNotFoundError,
)
from daita.loop.models import LoopBudgets, LoopState
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    TriggerKind,
)
from daita.operations.store import (
    CommitResult,
    OperationNotFoundError,
    OperationRevisionConflict,
)
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import SQLiteCorruptionError, SQLiteOperationStore

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)


def _snapshot(
    *,
    agent_id: str,
    operation_id: str,
    event_types: tuple[str, ...] = ("operation.created",),
    event_times: tuple[datetime, ...] | None = None,
) -> OperationSnapshot:
    if event_times is None:
        event_times = tuple(
            NOW + timedelta(microseconds=index) for index in range(len(event_types))
        )
    assert len(event_times) == len(event_types)
    trigger_id = f"{operation_id}:trigger"
    events = tuple(
        RuntimeEvent(
            id=f"{operation_id}:event:{index}:{event_type}",
            type=event_type,
            agent_id=agent_id,
            operation_id=operation_id,
            created_at=created_at,
            payload={"index": index, "operation_id": operation_id},
        )
        for index, (event_type, created_at) in enumerate(
            zip(event_types, event_times, strict=True)
        )
    )
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
            updated_at=max(event_times, default=NOW),
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
    created_at: datetime,
) -> OperationSnapshot:
    event = RuntimeEvent(
        id=f"{snapshot.operation.id}:event:{len(snapshot.events)}:{event_type}",
        type=event_type,
        agent_id=snapshot.operation.agent_id,
        operation_id=snapshot.operation.id,
        created_at=created_at,
        payload={"revision_event": len(snapshot.events)},
    )
    return replace(
        snapshot,
        operation=replace(snapshot.operation, updated_at=created_at),
        events=(*snapshot.events, event),
    )


def _raw_event_rows(path: Path) -> tuple[tuple[object, ...], ...]:
    connection = sqlite3.connect(path)
    try:
        return tuple(
            connection.execute(
                "SELECT agent_id, agent_sequence, id, type "
                "FROM runtime_events ORDER BY rowid"
            ).fetchall()
        )
    finally:
        connection.close()


def _install_abort_trigger(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("""
            CREATE TRIGGER contract_abort_committed_event
            BEFORE INSERT ON runtime_events
            WHEN NEW.type = 'contract.force_abort'
            BEGIN
                SELECT RAISE(ABORT, 'forced committed event failure');
            END
            """)
        connection.commit()
    finally:
        connection.close()


def _drop_abort_trigger(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute("DROP TRIGGER contract_abort_committed_event")
        connection.commit()
    finally:
        connection.close()


async def test_migration_three_backfills_agent_sequences_in_rowid_order(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    v2_migrations = sqlite_owner._MIGRATIONS[:2]
    assert tuple(migration.version for migration in v2_migrations) == (1, 2)
    v2_store = await sqlite_owner._open_with_migrations(
        path,
        migrations=v2_migrations,
        verify_owned_schema=True,
    )
    await v2_store.close()

    connection = sqlite3.connect(path)
    try:
        legacy_rows = (
            ("legacy-a-1", "a.first-inserted", "agent-a", NOW + timedelta(minutes=30)),
            ("legacy-a-2", "a.second-inserted", "agent-a", NOW + timedelta(minutes=20)),
            ("legacy-b-1", "b.first-inserted", "agent-b", NOW + timedelta(minutes=10)),
            ("legacy-a-3", "a.third-inserted", "agent-a", NOW),
        )
        connection.executemany(
            "INSERT INTO runtime_events("
            "id, operation_id, position, type, agent_id, created_at, payload_json"
            ") VALUES (?, NULL, NULL, ?, ?, ?, '{}')",
            (
                (
                    event_id,
                    event_type,
                    agent_id,
                    created_at.isoformat(timespec="microseconds").replace(
                        "+00:00", "Z"
                    ),
                )
                for event_id, event_type, agent_id, created_at in legacy_rows
            ),
        )
        connection.commit()
    finally:
        connection.close()

    upgraded = await SQLiteOperationStore.open(path)
    await upgraded.close()

    backup_path = path.with_name(f"{path.name}.before-v2.bak")
    assert backup_path.is_file()
    backup = sqlite3.connect(backup_path)
    try:
        assert backup.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,)]
        assert "agent_sequence" not in {
            str(row[1]) for row in backup.execute("PRAGMA table_info(runtime_events)")
        }
        assert backup.execute(
            "SELECT agent_id, type FROM runtime_events ORDER BY rowid"
        ).fetchall() == [
            ("agent-a", "a.first-inserted"),
            ("agent-a", "a.second-inserted"),
            ("agent-b", "b.first-inserted"),
            ("agent-a", "a.third-inserted"),
        ]
    finally:
        backup.close()

    connection = sqlite3.connect(path)
    try:
        versions = tuple(
            int(row[0])
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        )
        assert versions == (1, 2, 3)
        columns = {
            str(row[1]): (str(row[2]), int(row[3]))
            for row in connection.execute("PRAGMA table_info(runtime_events)")
        }
        assert columns["agent_sequence"] == ("INTEGER", 1)
        assert connection.execute(
            "SELECT agent_id, agent_sequence, type "
            "FROM runtime_events ORDER BY rowid"
        ).fetchall() == [
            ("agent-a", 1, "a.first-inserted"),
            ("agent-a", 2, "a.second-inserted"),
            ("agent-b", 1, "b.first-inserted"),
            ("agent-a", 3, "a.third-inserted"),
        ]
    finally:
        connection.close()


async def test_migration_three_copy_failure_preserves_version_two_atomically(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    v2_store = await sqlite_owner._open_with_migrations(
        path,
        migrations=sqlite_owner._MIGRATIONS[:2],
        verify_owned_schema=True,
    )
    await v2_store.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone() == (0,)
        connection.execute(
            "INSERT INTO runtime_events("
            "id, operation_id, position, type, agent_id, created_at, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?, '{}')",
            (
                "legacy-orphan",
                "missing-operation",
                0,
                "legacy.orphan",
                "agent-migration-failure",
                "2026-07-17T12:00:00.000000Z",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(
        sqlite_owner.SQLiteMigrationError,
        match=r"migration 3 \(project_committed_event_cursors\) failed",
    ):
        await SQLiteOperationStore.open(path)

    connection = sqlite3.connect(path)
    try:
        assert connection.execute("PRAGMA user_version").fetchone() == (2,)
        assert connection.execute(
            "SELECT version FROM schema_migrations ORDER BY version"
        ).fetchall() == [(1,), (2,)]
        assert "agent_sequence" not in {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(runtime_events)")
        }
        assert connection.execute(
            "SELECT id, operation_id FROM runtime_events"
        ).fetchall() == [("legacy-orphan", "missing-operation")]
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master "
                "WHERE name IN ('runtime_events_v3', "
                "'runtime_events_reject_update', 'runtime_events_reject_delete')"
            ).fetchall()
            == []
        )
    finally:
        connection.close()


async def test_migration_three_enforces_positive_unique_append_only_sequences(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(
            _snapshot(agent_id="agent-schema", operation_id="operation-schema")
        )
    finally:
        await store.close()

    connection = sqlite3.connect(path)
    try:
        unique_column_sets: set[tuple[str, ...]] = set()
        for index_row in connection.execute("PRAGMA index_list(runtime_events)"):
            if int(index_row[2]) != 1:
                continue
            index_name = str(index_row[1]).replace('"', '""')
            unique_column_sets.add(
                tuple(
                    str(column[2])
                    for column in connection.execute(
                        f'PRAGMA index_info("{index_name}")'
                    )
                )
            )
        assert ("agent_id", "agent_sequence") in unique_column_sets
        operation_foreign_keys = [
            row
            for row in connection.execute("PRAGMA foreign_key_list(runtime_events)")
            if str(row[2]) == "operations"
        ]
        assert len(operation_foreign_keys) == 1
        assert str(operation_foreign_keys[0][6]).upper() == "NO ACTION"

        raw_insert = (
            "INSERT INTO runtime_events("
            "id, operation_id, position, type, agent_id, agent_sequence, "
            "created_at, payload_json"
            ") VALUES (?, NULL, NULL, ?, ?, ?, ?, '{}')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                raw_insert,
                (
                    "event-zero",
                    "raw.zero",
                    "agent-zero",
                    0,
                    "2026-07-17T12:00:00.000000Z",
                ),
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                raw_insert,
                (
                    "event-duplicate-sequence",
                    "raw.duplicate",
                    "agent-schema",
                    1,
                    "2026-07-17T12:00:00.000000Z",
                ),
            )
        connection.rollback()

        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "UPDATE runtime_events SET payload_json = '{}' "
                "WHERE agent_id = 'agent-schema'"
            )
        connection.rollback()
        with pytest.raises(sqlite3.IntegrityError):
            connection.execute(
                "DELETE FROM runtime_events WHERE agent_id = 'agent-schema'"
            )
        connection.rollback()
        assert connection.execute(
            "SELECT COUNT(*) FROM runtime_events WHERE agent_id = 'agent-schema'"
        ).fetchone() == (1,)
    finally:
        connection.close()


async def test_read_after_paginates_from_start_and_rejects_cross_agent_cursor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    initial = _snapshot(
        agent_id="agent-reader",
        operation_id="operation-reader",
        event_types=("reader.one", "reader.two"),
    )
    third = _append_event(
        initial,
        event_type="reader.three",
        created_at=NOW + timedelta(seconds=1),
    )
    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(initial)
        await store.commit(third, expected_revision=1)

        first_page = await store.read_after("agent-reader", None, limit=2)
        assert first_page == (
            CommittedEvent(EventCursor("agent-reader", 1), initial.events[0]),
            CommittedEvent(EventCursor("agent-reader", 2), initial.events[1]),
        )
        second_page = await store.read_after(
            "agent-reader", first_page[-1].cursor, limit=2
        )
        assert second_page == (
            CommittedEvent(EventCursor("agent-reader", 3), third.events[-1]),
        )
        assert (
            await store.read_after("agent-reader", second_page[-1].cursor, limit=2)
            == ()
        )
        assert await store.read_after("agent-with-no-events", None, limit=10) == ()

        foreign_cursor = EventCursor("another-agent", 1)
        with pytest.raises(EventCursorMismatchError) as mismatch:
            await store.read_after("agent-reader", foreign_cursor, limit=1)
        assert mismatch.value.requested_agent_id == "agent-reader"
        assert mismatch.value.cursor == foreign_cursor

        unknown = EventCursor("agent-reader", 99)
        with pytest.raises(EventCursorNotFoundError) as missing:
            await store.read_after("agent-reader", unknown, limit=1)
        assert missing.value.cursor == unknown

        for invalid_limit in (0, -1, True, 1_001):
            with pytest.raises(ValueError, match="limit"):
                await store.read_after(
                    "agent-reader",
                    None,
                    limit=invalid_limit,
                )
    finally:
        await store.close()


async def test_create_and_commit_atomically_project_state_and_events(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    initial = _snapshot(agent_id="agent-atomic", operation_id="operation-atomic")
    candidate = _append_event(
        initial,
        event_type="operation.checkpointed",
        created_at=NOW + timedelta(seconds=1),
    )
    store = await SQLiteOperationStore.open(path)
    try:
        created = await store.create(initial)
        assert created.operation.revision == 1
        assert await store.read_after("agent-atomic", None, limit=10) == (
            CommittedEvent(EventCursor("agent-atomic", 1), initial.events[0]),
        )

        committed = await store.commit(candidate, expected_revision=1)
        assert committed.operation.revision == 2
        assert (await store.load(initial.operation.id)).snapshot == candidate
        assert await store.read_after("agent-atomic", None, limit=10) == (
            CommittedEvent(EventCursor("agent-atomic", 1), initial.events[0]),
            CommittedEvent(EventCursor("agent-atomic", 2), candidate.events[-1]),
        )
    finally:
        await store.close()


async def test_failed_create_and_commit_roll_back_without_sequence_gaps(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    rejected_create = _snapshot(
        agent_id="agent-rollback",
        operation_id="operation-rejected-create",
        event_types=("contract.force_abort",),
    )
    initial = _snapshot(
        agent_id="agent-rollback",
        operation_id="operation-rollback",
    )
    rejected_commit = _append_event(
        initial,
        event_type="contract.force_abort",
        created_at=NOW + timedelta(seconds=1),
    )
    accepted_commit = _append_event(
        initial,
        event_type="operation.accepted",
        created_at=NOW + timedelta(seconds=2),
    )

    store = await SQLiteOperationStore.open(path)
    try:
        _install_abort_trigger(path)
        with pytest.raises(sqlite3.IntegrityError, match="forced committed event"):
            await store.create(rejected_create)
        with pytest.raises(OperationNotFoundError):
            await store.load(rejected_create.operation.id)
        assert await store.read_after("agent-rollback", None, limit=10) == ()

        _drop_abort_trigger(path)
        await store.create(initial)
        _install_abort_trigger(path)
        with pytest.raises(sqlite3.IntegrityError, match="forced committed event"):
            await store.commit(rejected_commit, expected_revision=1)
        assert (await store.load(initial.operation.id)).revision == 1
        assert await store.read_after("agent-rollback", None, limit=10) == (
            CommittedEvent(EventCursor("agent-rollback", 1), initial.events[0]),
        )

        _drop_abort_trigger(path)
        committed = await store.commit(accepted_commit, expected_revision=1)
        assert committed.operation.revision == 2
        assert tuple(
            item.cursor.sequence
            for item in await store.read_after("agent-rollback", None, limit=10)
        ) == (1, 2)
    finally:
        connection = sqlite3.connect(path)
        try:
            trigger_exists = connection.execute(
                "SELECT 1 FROM sqlite_master "
                "WHERE type = 'trigger' AND name = 'contract_abort_committed_event'"
            ).fetchone()
        finally:
            connection.close()
        if trigger_exists is not None:
            _drop_abort_trigger(path)
        await store.close()


async def test_cas_loser_emits_no_event_and_reserves_no_sequence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    initial = _snapshot(agent_id="agent-cas", operation_id="operation-cas")
    candidates = (
        _append_event(
            initial,
            event_type="writer.a",
            created_at=NOW + timedelta(seconds=1),
        ),
        _append_event(
            initial,
            event_type="writer.b",
            created_at=NOW + timedelta(seconds=1),
        ),
    )
    first = await SQLiteOperationStore.open(path)
    second = await SQLiteOperationStore.open(path)
    try:
        await first.create(initial)
        outcomes = await asyncio.gather(
            first.commit(candidates[0], expected_revision=1),
            second.commit(candidates[1], expected_revision=1),
            return_exceptions=True,
        )
        winners = [result for result in outcomes if isinstance(result, CommitResult)]
        losers = [
            result
            for result in outcomes
            if isinstance(result, OperationRevisionConflict)
        ]
        assert len(winners) == 1
        assert len(losers) == 1

        projected = await first.read_after("agent-cas", None, limit=10)
        assert tuple(item.cursor.sequence for item in projected) == (1, 2)
        assert tuple(item.event for item in projected) == (
            initial.events[0],
            winners[0].committed_events[0],
        )
        assert len(_raw_event_rows(path)) == 2
    finally:
        await asyncio.gather(first.close(), second.close())


async def test_sequences_span_operations_are_agent_scoped_and_survive_reopen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    snapshots = (
        _snapshot(agent_id="agent-a", operation_id="operation-a-first"),
        _snapshot(agent_id="agent-b", operation_id="operation-b-first"),
        _snapshot(agent_id="agent-a", operation_id="operation-a-second"),
    )
    store = await SQLiteOperationStore.open(path)
    try:
        for snapshot in snapshots:
            await store.create(snapshot)
        assert tuple(
            item.cursor.sequence
            for item in await store.read_after("agent-a", None, limit=10)
        ) == (1, 2)
        assert tuple(
            item.cursor.sequence
            for item in await store.read_after("agent-b", None, limit=10)
        ) == (1,)
    finally:
        await store.close()

    reopened = await SQLiteOperationStore.open(path)
    try:
        agent_a = await reopened.read_after("agent-a", None, limit=10)
        agent_b = await reopened.read_after("agent-b", None, limit=10)
        assert tuple(item.event for item in agent_a) == (
            snapshots[0].events[0],
            snapshots[2].events[0],
        )
        assert tuple(item.event for item in agent_b) == (snapshots[1].events[0],)
        assert tuple(item.cursor for item in agent_a) == (
            EventCursor("agent-a", 1),
            EventCursor("agent-a", 2),
        )
        assert tuple(item.cursor for item in agent_b) == (EventCursor("agent-b", 1),)
    finally:
        await reopened.close()


@pytest.mark.parametrize(
    "mutation",
    (
        "UPDATE runtime_events SET agent_sequence = 5",
        "UPDATE runtime_events SET payload_json = CAST(payload_json AS BLOB)",
    ),
    ids=("sequence-gap", "wrong-storage-class"),
)
async def test_read_after_normalizes_corrupt_projection_rows(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteOperationStore.open(path)
    try:
        await store.create(
            _snapshot(agent_id="agent-corrupt", operation_id="operation-corrupt")
        )
        connection = sqlite3.connect(path)
        try:
            connection.execute("DROP TRIGGER runtime_events_reject_update")
            connection.execute(mutation)
            connection.commit()
        finally:
            connection.close()

        with pytest.raises(SQLiteCorruptionError):
            await store.read_after("agent-corrupt", None, limit=10)
    finally:
        await store.close()
