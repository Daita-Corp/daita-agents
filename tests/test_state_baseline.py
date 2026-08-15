from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from daita.errors import StateCompatibilityCode, StateCompatibilityError
from daita.llm.models import ModelUsage
from daita.llm.pricing import CostEstimate, CostEstimateStatus
from daita.loop.models import LoopExit, LoopExitKind, RunInput
from daita.storage import sqlite as sqlite_store
from daita.storage.sqlite import (
    DatabaseWriteReceipt,
    SQLiteStateStore,
)
from daita.storage.sqlite_codecs import (
    decode_loop_exit,
    encode_loop_exit,
    encode_receipt,
    encode_run_input,
)
from daita.storage.sqlite_migrations import CURRENT_REVISION, migration_rows
from daita.storage.sqlite_migrations import runner as migration_runner
from daita.storage.sqlite_migrations.generalized_postgresql_updates import (
    MIGRATION as GENERALIZED_UPDATE_MIGRATION,
)
from daita.storage.sqlite_schema import CURRENT_TABLES, table_names


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _journal(path: Path) -> tuple[tuple[int, str, str], ...]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(
                "SELECT ordinal, migration_id, checksum "
                "FROM state_migrations ORDER BY ordinal"
            )
        )


def _legacy_receipt() -> DatabaseWriteReceipt:
    return DatabaseWriteReceipt.start(
        agent_id="agent-upgrade",
        run_id="run-upgrade",
        call_id="call-upgrade",
        capability_id="data.postgresql.update",
        source_id="source:sha256:" + "1" * 64,
        resource_id="catalog-resource:sha256:" + "2" * 64,
        intent_sha256="sha256:" + "3" * 64,
        preview_fingerprint="sha256:" + "4" * 64,
        expected_affected_rows=1,
        started_at=datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
    )


def _make_three_migration_state(
    path: Path,
    receipt: DatabaseWriteReceipt,
) -> tuple[str, str]:
    encoded = json.loads(encode_receipt(receipt))
    del encoded["fields"]["expected_affected_rows"]
    legacy_data = json.dumps(encoded, sort_keys=True, separators=(",", ":"))
    run_input = RunInput(
        id="run-legacy-cost",
        agent_id="agent-upgrade",
        message="Preserve this completed run.",
        conversation_id="conversation-upgrade",
        created_at=datetime(2026, 8, 12, 12, 1, tzinfo=UTC),
    )
    loop_exit = LoopExit(
        run_id=run_input.id,
        conversation_id="conversation-upgrade",
        kind=LoopExitKind.COMPLETED,
        reason="completed",
        created_at=datetime(2026, 8, 12, 12, 2, tzinfo=UTC),
        final_text="Preserved answer.",
        usage=ModelUsage(
            input_tokens=12,
            output_tokens=4,
            cost_estimate=CostEstimate.complete(Decimal("0.0125")),
        ),
    )
    legacy_result = json.loads(encode_loop_exit(loop_exit))
    usage_fields = legacy_result["fields"]["usage"]["fields"]
    cost_fields = usage_fields.pop("cost_estimate")["fields"]
    usage_fields["estimated_cost_usd"] = cost_fields["amount_usd"]
    legacy_result_data = json.dumps(
        legacy_result,
        sort_keys=True,
        separators=(",", ":"),
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM state_migrations WHERE ordinal = 4")
        connection.execute(
            """INSERT INTO database_write_receipts
               (agent_id, id, run_id, call_id, data)
               VALUES (?, ?, ?, ?, ?)""",
            (
                receipt.agent_id,
                receipt.receipt_id,
                receipt.run_id,
                receipt.call_id,
                legacy_data,
            ),
        )
        connection.execute(
            """INSERT INTO runs
               (id, agent_id, conversation_id, turn_index, input, result)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                run_input.id,
                run_input.agent_id,
                run_input.conversation_id,
                0,
                encode_run_input(run_input),
                legacy_result_data,
            ),
        )
    return legacy_data, legacy_result_data


async def test_fresh_state_is_created_directly_at_current_append_only_revision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()

    with sqlite3.connect(path) as connection:
        assert table_names(connection) == set(CURRENT_TABLES)
    assert _journal(path) == migration_rows()
    assert len(migration_rows()) == 4
    assert migration_rows()[-1][1] == CURRENT_REVISION


async def test_known_prefix_upgrades_on_staged_copy_and_retains_exact_rollback(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    receipt = _legacy_receipt()
    legacy_data, legacy_result_data = _make_three_migration_state(path, receipt)
    before_rows: dict[str, tuple[tuple[object, ...], ...]]
    with sqlite3.connect(path) as connection:
        before_rows = {
            table: tuple(connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
            for table in table_names(connection)
            if table not in {"database_write_receipts", "runs", "state_migrations"}
        }

    upgraded = await SQLiteStateStore.open(path)
    await upgraded.close()

    assert _journal(path) == migration_rows()
    with sqlite3.connect(path) as connection:
        data = connection.execute(
            "SELECT data FROM database_write_receipts"
        ).fetchone()[0]
        assert json.loads(data)["fields"]["expected_affected_rows"] == 1
        assert {
            table: tuple(connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
            for table in before_rows
        } == before_rows
        migrated_result = connection.execute(
            "SELECT result FROM runs WHERE id = 'run-legacy-cost'"
        ).fetchone()[0]
        exit_record = decode_loop_exit(migrated_result)
        assert exit_record.usage.cost_estimate.status is CostEstimateStatus.PARTIAL
        assert exit_record.usage.cost_estimate.amount_usd == Decimal("0.0125")
        assert (
            exit_record.usage.cost_estimate.code
            == "legacy_estimate_completeness_unknown"
        )

    rollback_points = tuple(tmp_path.glob("state.db.rollback-*"))
    assert len(rollback_points) == 1
    rollback = rollback_points[0]
    assert _journal(rollback) == migration_rows()[:3]
    with sqlite3.connect(rollback) as connection:
        assert connection.execute(
            "SELECT data FROM database_write_receipts"
        ).fetchone() == (legacy_data,)
        assert connection.execute(
            "SELECT result FROM runs WHERE id = 'run-legacy-cost'"
        ).fetchone() == (legacy_result_data,)


async def test_upgrade_preserves_unreadable_run_result_byte_exact(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    _make_three_migration_state(path, _legacy_receipt())
    with sqlite3.connect(path) as connection:
        result = connection.execute(
            "SELECT result FROM runs WHERE id = 'run-legacy-cost'"
        ).fetchone()[0]
        payload = json.loads(result)
        del payload["fields"]["usage"]["fields"]["cache_write_tokens"]
        unreadable = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        connection.execute(
            "UPDATE runs SET result = ? WHERE id = 'run-legacy-cost'",
            (unreadable,),
        )

    upgraded = await SQLiteStateStore.open(path)
    await upgraded.close()

    with sqlite3.connect(path) as connection:
        migrated = connection.execute(
            "SELECT result FROM runs WHERE id = 'run-legacy-cost'"
        ).fetchone()[0]
    assert migrated == unreadable
    with pytest.raises(ValueError):
        decode_loop_exit(migrated)


async def test_failed_staged_migration_leaves_live_database_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    _make_three_migration_state(path, _legacy_receipt())
    before = _sha256(path)

    def fail_after_rewrite(connection: sqlite3.Connection) -> None:
        GENERALIZED_UPDATE_MIGRATION.apply(connection)
        raise RuntimeError("controlled staged migration failure")

    controlled = replace(GENERALIZED_UPDATE_MIGRATION, apply=fail_after_rewrite)
    monkeypatch.setattr(
        migration_runner,
        "MIGRATIONS",
        (*migration_runner.MIGRATIONS[:3], controlled),
    )
    monkeypatch.setattr(
        sqlite_store,
        "MIGRATIONS",
        (*sqlite_store.MIGRATIONS[:3], controlled),
    )

    with pytest.raises(StateCompatibilityError) as captured:
        await SQLiteStateStore.open(path)

    assert captured.value.code is StateCompatibilityCode.UPGRADE_FAILED
    assert captured.value.to_mapping()["state_changed"] is False
    assert _sha256(path) == before
    assert tuple(tmp_path.glob("state.db.rollback-*")) == ()
    assert tuple(tmp_path.glob(".state.db.*.db")) == ()


async def test_unknown_checksum_fails_closed_without_creating_a_backup(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await SQLiteStateStore.open(path)
    await store.close()
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE state_migrations SET checksum = ?", ("0" * 64,))
    before = _sha256(path)

    with pytest.raises(StateCompatibilityError) as captured:
        await SQLiteStateStore.open(path)

    assert captured.value.code is StateCompatibilityCode.REVISION_UNSUPPORTED
    assert _sha256(path) == before
    assert tuple(tmp_path.glob("state.db.rollback-*")) == ()
