from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from daita.storage.sqlite import (
    DatabaseWriteOutcome,
    DatabaseWriteReceipt,
    DatabaseWriteReceiptConflictError,
    SQLiteStateStore,
)

STARTED_AT = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
COMPLETED_AT = STARTED_AT + timedelta(seconds=1)


def _started(call_id: str = "call-one") -> DatabaseWriteReceipt:
    return DatabaseWriteReceipt.start(
        agent_id="agent-database-write",
        run_id="run-database-write",
        call_id=call_id,
        capability_id="data.postgresql.update",
        source_id="source:sha256:" + "1" * 64,
        resource_id="catalog-resource:sha256:" + "2" * 64,
        intent_sha256="sha256:" + "3" * 64,
        preview_fingerprint="sha256:" + "4" * 64,
        expected_affected_rows=7,
        started_at=STARTED_AT,
    )


@pytest.mark.parametrize(
    ("outcome", "affected_rows", "error_code"),
    (
        (DatabaseWriteOutcome.COMMITTED, 7, None),
        (DatabaseWriteOutcome.NOT_COMMITTED, 0, "write_constraint_violation"),
        (DatabaseWriteOutcome.OUTCOME_UNKNOWN, None, "write_outcome_unknown"),
    ),
)
async def test_receipt_state_machine_persists_each_terminal_outcome_immutably(
    tmp_path,
    outcome: DatabaseWriteOutcome,
    affected_rows: int | None,
    error_code: str | None,
):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: STARTED_AT)
    started = _started()
    try:
        assert await store.start_database_write_receipt(started) == started
        assert (
            await store.load_database_write_receipt(
                started.agent_id, started.receipt_id
            )
            == started
        )
        assert (
            await store.load_database_write_receipt_for_call(
                started.agent_id, started.run_id, started.call_id
            )
            == started
        )

        terminal = started.finish(
            outcome,
            completed_at=COMPLETED_AT,
            affected_rows=affected_rows,
            normalized_error_code=error_code,
        )
        assert await store.finish_database_write_receipt(terminal) == terminal
        assert await store.finish_database_write_receipt(terminal) == terminal

        conflicting = started.finish(
            DatabaseWriteOutcome.OUTCOME_UNKNOWN,
            completed_at=COMPLETED_AT + timedelta(seconds=1),
            affected_rows=None,
            normalized_error_code="write_outcome_unknown",
        )
        if conflicting != terminal:
            with pytest.raises(
                DatabaseWriteReceiptConflictError,
                match="terminal receipt is immutable",
            ):
                await store.finish_database_write_receipt(conflicting)
    finally:
        await store.close()


async def test_receipts_reject_invalid_or_repeated_execution_identity(tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: STARTED_AT)
    started = _started()
    try:
        with pytest.raises(ValueError, match="started receipt"):
            await store.finish_database_write_receipt(started)

        terminal = started.finish(
            DatabaseWriteOutcome.NOT_COMMITTED,
            completed_at=COMPLETED_AT,
            affected_rows=0,
            normalized_error_code="write_not_committed",
        )
        with pytest.raises(DatabaseWriteReceiptConflictError, match="does not exist"):
            await store.finish_database_write_receipt(terminal)

        await store.start_database_write_receipt(started)
        with pytest.raises(
            DatabaseWriteReceiptConflictError,
            match="execution identity already has a receipt",
        ):
            await store.start_database_write_receipt(started)

        different_intent = DatabaseWriteReceipt.start(
            agent_id=started.agent_id,
            run_id=started.run_id,
            call_id=started.call_id,
            capability_id=started.capability_id,
            source_id=started.source_id,
            resource_id=started.resource_id,
            intent_sha256="sha256:" + "9" * 64,
            preview_fingerprint=started.preview_fingerprint,
            expected_affected_rows=started.expected_affected_rows,
            started_at=started.started_at,
        )
        with pytest.raises(
            DatabaseWriteReceiptConflictError,
            match="execution identity already has a receipt",
        ):
            await store.start_database_write_receipt(different_intent)
    finally:
        await store.close()


async def test_leftover_started_receipt_recovers_to_unknown_on_reopen(tmp_path):
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path, clock=lambda: STARTED_AT)
    started = _started()
    await store.start_database_write_receipt(started)
    await store.close()

    reopened = await SQLiteStateStore.open(path, clock=lambda: COMPLETED_AT)
    try:
        recovered = await reopened.load_database_write_receipt(
            started.agent_id, started.receipt_id
        )
        assert recovered is not None
        assert recovered.outcome is DatabaseWriteOutcome.OUTCOME_UNKNOWN
        assert recovered.affected_rows is None
        assert recovered.normalized_error_code == "write_outcome_unknown"
        assert recovered.started_at == STARTED_AT
        assert recovered.completed_at == COMPLETED_AT
    finally:
        await reopened.close()


async def test_receipt_payload_contains_only_bounded_metadata(tmp_path):
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path, clock=lambda: STARTED_AT)
    started = _started()
    try:
        await store.start_database_write_receipt(started)
    finally:
        await store.close()

    with sqlite3.connect(path) as connection:
        payload = connection.execute(
            "SELECT data FROM database_write_receipts WHERE agent_id = ? AND id = ?",
            (started.agent_id, started.receipt_id),
        ).fetchone()[0]
    fields = json.loads(payload)["fields"]
    assert set(fields) == {
        "affected_rows",
        "agent_id",
        "call_id",
        "capability_id",
        "completed_at",
        "intent_sha256",
        "expected_affected_rows",
        "normalized_error_code",
        "outcome",
        "preview_fingerprint",
        "receipt_id",
        "resource_id",
        "run_id",
        "source_id",
        "started_at",
    }
    assert "sql" not in fields
    assert "password" not in fields
    assert "match" not in fields
    assert "assignments" not in fields
    assert "secret-value" not in payload
