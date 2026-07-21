from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3

import pytest

from daita._json import canonical_json
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
from daita.operations.models import (
    AgentTrigger,
    Operation,
    OperationStatus,
    Task,
    TaskStatus,
    TriggerKind,
)
from daita.storage import sqlite as sqlite_owner
from daita.storage.sqlite import SQLiteMigrationError, SQLiteOperationStore

NOW = datetime(2026, 7, 17, 20, 0, tzinfo=timezone.utc)
ZERO_SHA256 = "sha256:" + ("0" * 64)
MIGRATION_FOUR_NAME = "normalize_fenced_task_execution"
MIGRATION_FIVE_NAME = "link_blob_backed_evidence"
MIGRATION_SIX_NAME = "normalize_approval_lifecycle"
MIGRATION_SEVEN_NAME = "add_agent_identity_and_sessions"
MIGRATION_THIRTEEN_NAME = "persist_task_validation_facts"
MIGRATION_FOURTEEN_NAME = "bind_agent_runtime_defaults"
LATEST_MIGRATION_NAME = "persist_wave1_runtime_foundation"
LEGACY_RECOVERY_REASON = "legacy_running_task_missing_lease"
LEGACY_RECOVERY_EVENT_TYPE = "task.manual_recovery_required"

TASK_EXECUTION_COLUMNS = {
    "capability_fingerprint": ("TEXT", 1),
    "arguments_hash": ("TEXT", 1),
    "access_mode": ("TEXT", 1),
    "risk": ("TEXT", 1),
    "side_effecting": ("INTEGER", 1),
    "idempotent": ("INTEGER", 1),
    "replay_safe": ("INTEGER", 1),
    "idempotency_key": ("TEXT", 0),
    "manual_recovery_reason": ("TEXT", 0),
    "validation_schema_version": ("INTEGER", 1),
    "validation_passed": ("INTEGER", 1),
    "validation_in_scope": ("INTEGER", 1),
    "validation_destructive": ("INTEGER", 1),
    "validation_sensitivity_class": ("TEXT", 1),
    "validation_source_id": ("TEXT", 0),
    "validation_resource_ids_json": ("TEXT", 1),
    "validation_resource_revisions_json": ("TEXT", 1),
    "validation_source_revision": ("TEXT", 0),
    "validation_impact_json": ("TEXT", 1),
    "validation_evidence_ids_json": ("TEXT", 1),
    "validation_source_ids_json": ("TEXT", 1),
    "validation_source_revisions_json": ("TEXT", 1),
    "validation_freshness_state": ("TEXT", 0),
}
TASK_DEPENDENCY_COLUMNS = (
    "operation_id",
    "position",
    "task_id",
    "prerequisite_task_id",
)
TASK_LEASE_COLUMNS = (
    "operation_id",
    "position",
    "task_id",
    "attempt",
    "fencing_token",
    "holder_id",
    "acquired_at",
    "expires_at",
    "started_at",
    "renewed_at",
    "released_at",
    "release_reason",
)
TASK_EXECUTION_INDEXES = {
    "task_dependencies_task_idx",
    "task_dependencies_prerequisite_idx",
    "task_leases_task_fence_idx",
    "task_leases_one_unreleased_idx",
}
TASK_EXECUTION_TRIGGERS = {
    "task_dependencies_reject_update",
    "task_dependencies_reject_delete",
    "task_leases_reject_identity_update",
    "task_leases_reject_delete",
}


def _migration_prefix() -> tuple[sqlite_owner._SQLiteMigration, ...]:
    migrations = tuple(sqlite_owner._MIGRATIONS)
    assert tuple(migration.version for migration in migrations[:3]) == (1, 2, 3)
    assert tuple(migration.name for migration in migrations[:3]) == (
        "initialize_sqlite_foundation",
        "normalize_operation_lifecycle",
        "project_committed_event_cursors",
    )
    return migrations[:3]


def _migration_four() -> sqlite_owner._SQLiteMigration:
    migrations = tuple(sqlite_owner._MIGRATIONS)
    assert tuple(migration.version for migration in migrations[:4]) == (1, 2, 3, 4)
    migration = migrations[3]
    assert migration.name == MIGRATION_FOUR_NAME
    return migration


def _migration_five() -> sqlite_owner._SQLiteMigration:
    migrations = tuple(sqlite_owner._MIGRATIONS)
    assert tuple(migration.version for migration in migrations[:5]) == (
        1,
        2,
        3,
        4,
        5,
    )
    migration = migrations[4]
    assert migration.name == MIGRATION_FIVE_NAME
    return migration


def _migration_six() -> sqlite_owner._SQLiteMigration:
    migrations = tuple(sqlite_owner._MIGRATIONS)
    assert tuple(migration.version for migration in migrations[:6]) == (
        1,
        2,
        3,
        4,
        5,
        6,
    )
    migration = migrations[5]
    assert migration.name == MIGRATION_SIX_NAME
    return migration


def _legacy_snapshot() -> OperationSnapshot:
    trigger = AgentTrigger(
        id="trigger-legacy",
        agent_id="agent-legacy",
        kind=TriggerKind.USER,
        source_id="user-legacy",
        payload={"message": "Run two legacy tasks."},
        created_at=NOW,
        session_id="session-legacy",
    )
    operation = Operation(
        id="operation-legacy",
        agent_id=trigger.agent_id,
        trigger_id=trigger.id,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
        session_id=trigger.session_id,
    )
    request = ModelRequest(
        operation_id=operation.id,
        turn_id="turn-legacy",
        messages=(
            CanonicalMessage(
                agent_id=operation.agent_id,
                operation_id=operation.id,
                turn_id="turn-legacy",
                session_id=operation.session_id,
                role=MessageRole.USER,
                content=(TextBlock("Run two legacy tasks."),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="legacy.read",
                description="Read a legacy value.",
                input_schema={"type": "object"},
            ),
        ),
    )
    response = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id="call-pending",
                name="legacy.read",
                arguments={"key": "pending"},
            ),
            ToolCall(
                id="call-running",
                name="legacy.read",
                arguments={"key": "running"},
            ),
        ),
    )
    model_call = ModelCall(
        id="model-call-legacy",
        operation_id=operation.id,
        turn_id="turn-legacy",
        provider_id="mock:legacy",
        request=request,
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
        response=response,
    )
    turn = Turn(
        id="turn-legacy",
        operation_id=operation.id,
        number=1,
        created_at=NOW,
        model_request_id=model_call.id,
        model_response_id=model_call.id,
    )
    tasks = tuple(
        Task(
            id=task_id,
            operation_id=operation.id,
            turn_id=turn.id,
            call_id=call_id,
            capability_id="legacy.read",
            executor_id="legacy.executor",
            status=status,
            attempt=1,
            arguments={"key": key},
            created_at=NOW,
            updated_at=NOW,
        )
        for task_id, call_id, key, status in (
            (
                "task-pending",
                "call-pending",
                "pending",
                TaskStatus.PENDING,
            ),
            (
                "task-running",
                "call-running",
                "running",
                TaskStatus.RUNNING,
            ),
        )
    )
    event = RuntimeEvent(
        id="event-operation-created-legacy",
        type="operation.created",
        agent_id=operation.agent_id,
        operation_id=operation.id,
        session_id=operation.session_id,
        payload={},
        created_at=NOW,
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.AWAITING_EXECUTION,
            turn_count=1,
            action_count=2,
        ),
        budgets=LoopBudgets(),
        turns=(turn,),
        model_calls=(model_call,),
        readiness=(),
        tasks=tasks,
        evidence=(),
        observations=(),
        events=(event,),
    )


async def _seed_version_three(path: Path) -> OperationSnapshot:
    snapshot = _legacy_snapshot()
    store = await sqlite_owner._open_with_migrations(
        path,
        migrations=_migration_prefix(),
        verify_owned_schema=True,
    )
    try:
        created = await store.create(replace(snapshot, tasks=()))
        assert created.operation.revision == 1
    finally:
        await store.close()

    encoded_now = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    connection = sqlite3.connect(path)
    try:
        connection.executemany(
            "INSERT INTO tasks("
            "operation_id, position, id, turn_id, call_id, capability_id, "
            "executor_id, status, attempt, arguments_json, created_at, "
            "updated_at, error_code, cancellation_requested"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(
                (
                    task.operation_id,
                    position,
                    task.id,
                    task.turn_id,
                    task.call_id,
                    task.capability_id,
                    task.executor_id,
                    task.status.value,
                    task.attempt,
                    canonical_json(task.arguments),
                    encoded_now,
                    encoded_now,
                    task.error_code,
                    int(task.cancellation_requested),
                )
                for position, task in enumerate(snapshot.tasks)
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return snapshot


async def _seed_version_four_with_evidence(
    path: Path,
) -> tuple[OperationSnapshot, str]:
    snapshot = await _seed_version_three(path)
    version_four = await sqlite_owner._open_with_migrations(
        path,
        migrations=(*_migration_prefix(), _migration_four()),
        verify_owned_schema=True,
    )
    await version_four.close()

    evidence_id = "evidence-version-four"
    encoded_now = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "INSERT INTO evidence("
            "operation_id, position, id, task_id, turn_id, capability_id, "
            "executor_id, kind, schema_version, attempt, accepted, payload_json, "
            "content_hash, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.operation.id,
                0,
                evidence_id,
                "task-pending",
                "turn-legacy",
                "legacy.read",
                "legacy.executor",
                "legacy.record",
                1,
                1,
                0,
                "{}",
                ZERO_SHA256,
                encoded_now,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return snapshot, evidence_id


def _logical_database_image(path: Path) -> tuple[int, int, tuple[str, ...]]:
    connection = sqlite3.connect(path)
    try:
        application_id = int(connection.execute("PRAGMA application_id").fetchone()[0])
        user_version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        return application_id, user_version, tuple(connection.iterdump())
    finally:
        connection.close()


def _schema_names(path: Path, schema_type: str) -> set[str]:
    connection = sqlite3.connect(path)
    try:
        return {
            str(row[0])
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type = ?",
                (schema_type,),
            )
            if not str(row[0]).startswith("sqlite_")
        }
    finally:
        connection.close()


def _columns(path: Path, table_name: str) -> dict[str, tuple[str, int]]:
    connection = sqlite3.connect(path)
    try:
        quoted_name = table_name.replace('"', '""')
        return {
            str(row[1]): (str(row[2]).upper(), int(row[3]))
            for row in connection.execute(f'PRAGMA table_info("{quoted_name}")')
        }
    finally:
        connection.close()


def _migration_rows(path: Path) -> tuple[tuple[int, str, str], ...]:
    connection = sqlite3.connect(path)
    try:
        return tuple(
            (int(row[0]), str(row[1]), str(row[2]))
            for row in connection.execute(
                "SELECT version, name, checksum "
                "FROM schema_migrations ORDER BY version"
            )
        )
    finally:
        connection.close()


def _foreign_key_targets(path: Path, table_name: str) -> tuple[str, ...]:
    connection = sqlite3.connect(path)
    try:
        quoted_name = table_name.replace('"', '""')
        rows = connection.execute(
            f'PRAGMA foreign_key_list("{quoted_name}")'
        ).fetchall()
        targets_by_constraint = {int(row[0]): str(row[2]) for row in rows}
        return tuple(sorted(targets_by_constraint.values()))
    finally:
        connection.close()


def _assert_task_execution_schema(path: Path) -> None:
    tables = _schema_names(path, "table")
    assert {"task_dependencies", "task_leases"}.issubset(tables)
    assert TASK_EXECUTION_COLUMNS.items() <= _columns(path, "tasks").items()
    assert tuple(_columns(path, "task_dependencies")) == TASK_DEPENDENCY_COLUMNS
    assert tuple(_columns(path, "task_leases")) == TASK_LEASE_COLUMNS
    assert TASK_EXECUTION_INDEXES.issubset(_schema_names(path, "index"))
    assert TASK_EXECUTION_TRIGGERS.issubset(_schema_names(path, "trigger"))
    assert _foreign_key_targets(path, "task_dependencies") == (
        "operations",
        "tasks",
        "tasks",
    )
    assert _foreign_key_targets(path, "task_leases") == ("operations", "tasks")


def _assert_evidence_blob_schema(path: Path) -> None:
    assert _columns(path, "evidence")["blob_id"] == ("TEXT", 0)


async def test_public_schema_migrations_are_fixed_and_reopen_is_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "fresh-v5.db"
    migration_five = _migration_five()
    normalized_statements = tuple(
        " ".join(statement.lower().split()) for statement in migration_five.statements
    )
    assert len(normalized_statements) == 1
    assert normalized_statements[0].startswith(
        "alter table evidence add column blob_id text"
    )
    assert "not null" not in normalized_statements[0]

    first = await SQLiteOperationStore.open(path)
    await first.close()

    assert _migration_rows(path) == tuple(
        (migration.version, migration.name, migration.checksum)
        for migration in sqlite_owner._MIGRATIONS
    )
    assert _migration_rows(path)[-1][1] == LATEST_MIGRATION_NAME
    _assert_task_execution_schema(path)
    _assert_evidence_blob_schema(path)
    first_image = _logical_database_image(path)

    reopened = await SQLiteOperationStore.open(path)
    await reopened.close()

    assert _logical_database_image(path) == first_image
    assert migration_five.version == 5
    assert _migration_six().version == 6


async def test_version_four_upgrade_adds_nullable_evidence_blob_id(
    tmp_path: Path,
) -> None:
    assert _migration_five().version == 5
    path = tmp_path / "legacy-v4.db"
    backup_path = tmp_path / "legacy-v4.exact-backup.db"
    legacy, evidence_id = await _seed_version_four_with_evidence(path)
    version_four_image = _logical_database_image(path)
    version_four_migrations = (*_migration_prefix(), _migration_four())

    upgraded = await SQLiteOperationStore.open(path, backup_path=backup_path)
    await upgraded.close()

    assert _logical_database_image(backup_path) == version_four_image
    assert _migration_rows(backup_path) == tuple(
        (migration.version, migration.name, migration.checksum)
        for migration in version_four_migrations
    )
    assert "blob_id" not in _columns(backup_path, "evidence")
    _assert_evidence_blob_schema(path)
    assert _migration_rows(path)[-1][1] == LATEST_MIGRATION_NAME

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT blob_id, metadata_schema_version, acceptance_reason, "
            "rejection_reason, "
            "applicable, applicability_reason, validation_facts_json, "
            "projection_metadata_json, redaction_metadata_json "
            "FROM evidence WHERE operation_id = ? AND id = ?",
            (legacy.operation.id, evidence_id),
        ).fetchone() == (None, 0, None, None, 0, None, "{}", "{}", "{}")
    finally:
        connection.close()

    verifier = await SQLiteOperationStore.open(path)
    try:
        loaded = await verifier.load(legacy.operation.id)
    finally:
        await verifier.close()
    persisted = next(
        item for item in loaded.snapshot.evidence if item.id == evidence_id
    )
    assert persisted.metadata_schema_version == 0
    assert persisted.validation_facts.schema_version == 0
    assert persisted.acceptance_reason is None
    assert persisted.rejection_reason is None
    assert persisted.applicability_reason is None
    assert persisted.applicable is False

    upgraded_image = _logical_database_image(path)
    reopened = await SQLiteOperationStore.open(path)
    await reopened.close()
    assert _logical_database_image(path) == upgraded_image


async def test_version_three_upgrade_backfills_fail_closed_and_is_reopen_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-v3.db"
    backup_path = tmp_path / "legacy-v3.exact-backup.db"
    legacy = await _seed_version_three(path)
    version_three_image = _logical_database_image(path)

    upgraded = await SQLiteOperationStore.open(path, backup_path=backup_path)
    try:
        loaded = await upgraded.load(legacy.operation.id)
    finally:
        await upgraded.close()

    assert backup_path.is_file()
    assert _logical_database_image(backup_path) == version_three_image
    assert _migration_rows(backup_path) == tuple(
        (migration.version, migration.name, migration.checksum)
        for migration in _migration_prefix()
    )
    _assert_task_execution_schema(path)
    _assert_evidence_blob_schema(path)

    connection = sqlite3.connect(path)
    try:
        connection.row_factory = sqlite3.Row
        task_rows = connection.execute(
            "SELECT id, status, capability_fingerprint, arguments_hash, "
            "access_mode, risk, side_effecting, idempotent, replay_safe, "
            "idempotency_key, manual_recovery_reason "
            "FROM tasks WHERE operation_id = ? ORDER BY position",
            (legacy.operation.id,),
        ).fetchall()
        assert [str(row["id"]) for row in task_rows] == [
            "task-pending",
            "task-running",
        ]
        for row in task_rows:
            assert row["capability_fingerprint"] == ZERO_SHA256
            assert row["arguments_hash"] == ZERO_SHA256
            assert row["access_mode"] == AccessMode.WRITE.value
            assert row["risk"] == RiskLevel.HIGH.value
            assert row["side_effecting"] == 1
            assert row["idempotent"] == 0
            assert row["replay_safe"] == 0
            assert row["idempotency_key"] is None

        assert task_rows[0]["status"] == TaskStatus.PENDING.value
        assert task_rows[0]["manual_recovery_reason"] is None
        assert task_rows[1]["status"] == TaskStatus.MANUAL_RECOVERY_REQUIRED.value
        assert task_rows[1]["manual_recovery_reason"] == LEGACY_RECOVERY_REASON

        revision_row = connection.execute(
            "SELECT revision FROM operations WHERE id = ?",
            (legacy.operation.id,),
        ).fetchone()
        assert revision_row is not None
        assert int(revision_row[0]) == 2
        recovery_rows = connection.execute(
            "SELECT type, agent_id, agent_sequence, operation_id, position, "
            "task_id, capability_id, executor_id, payload_json "
            "FROM runtime_events WHERE operation_id = ? AND type = ?",
            (legacy.operation.id, LEGACY_RECOVERY_EVENT_TYPE),
        ).fetchall()
        assert len(recovery_rows) == 1
        recovery = recovery_rows[0]
        assert recovery["agent_id"] == legacy.operation.agent_id
        assert recovery["agent_sequence"] == 2
        assert recovery["operation_id"] == legacy.operation.id
        assert recovery["position"] == 1
        assert recovery["task_id"] == "task-running"
        assert recovery["capability_id"] == "legacy.read"
        assert recovery["executor_id"] == "legacy.executor"
        payload = json.loads(str(recovery["payload_json"]))
        assert payload == {
            "from_status": TaskStatus.RUNNING.value,
            "reason": LEGACY_RECOVERY_REASON,
            "to_status": TaskStatus.MANUAL_RECOVERY_REQUIRED.value,
        }
        dependency_count = connection.execute(
            "SELECT COUNT(*) FROM task_dependencies"
        ).fetchone()
        lease_count = connection.execute("SELECT COUNT(*) FROM task_leases").fetchone()
        assert dependency_count is not None and int(dependency_count[0]) == 0
        assert lease_count is not None and int(lease_count[0]) == 0
    finally:
        connection.close()

    pending, recovered = loaded.snapshot.tasks
    assert pending.status is TaskStatus.PENDING
    assert recovered.status is TaskStatus.MANUAL_RECOVERY_REQUIRED
    assert recovered.manual_recovery_reason == LEGACY_RECOVERY_REASON
    for task in (pending, recovered):
        assert task.execution_facts.capability_fingerprint == ZERO_SHA256
        assert task.execution_facts.arguments_hash == ZERO_SHA256
        assert task.execution_facts.access_mode is AccessMode.WRITE
        assert task.execution_facts.risk is RiskLevel.HIGH
        assert task.execution_facts.side_effecting is True
        assert task.execution_facts.idempotent is False
        assert task.execution_facts.replay_safe is False
        assert task.execution_facts.idempotency_key is None
        assert task.execution_facts.validation_facts.schema_version == 0
        assert task.execution_facts.validation_facts.fingerprint is None
    recovery_event = loaded.snapshot.events[-1]
    assert recovery_event.type == LEGACY_RECOVERY_EVENT_TYPE
    assert recovery_event.task_id == recovered.id

    upgraded_image = _logical_database_image(path)
    reopened = await SQLiteOperationStore.open(path)
    try:
        reloaded = await reopened.load(legacy.operation.id)
    finally:
        await reopened.close()
    assert _logical_database_image(path) == upgraded_image
    assert reloaded == loaded


async def test_version_three_recovery_appends_after_nontrivial_checkpoint(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-nontrivial-v3.db"
    legacy = await _seed_version_three(path)
    encoded_now = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            "UPDATE tasks SET status = ? WHERE id = ?",
            (TaskStatus.RUNNING.value, "task-pending"),
        )
        connection.execute(
            "UPDATE operations SET revision = 7 WHERE id = ?",
            (legacy.operation.id,),
        )
        connection.execute(
            "INSERT INTO runtime_events("
            "id, operation_id, position, type, agent_id, agent_sequence, "
            "created_at, session_id, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "event-legacy-checkpoint",
                legacy.operation.id,
                1,
                "legacy.checkpoint",
                legacy.operation.agent_id,
                2,
                encoded_now,
                legacy.operation.session_id,
                "{}",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    upgraded = await SQLiteOperationStore.open(path)
    try:
        loaded = await upgraded.load(legacy.operation.id)
    finally:
        await upgraded.close()

    assert loaded.revision == 8
    assert [task.status for task in loaded.snapshot.tasks] == [
        TaskStatus.MANUAL_RECOVERY_REQUIRED,
        TaskStatus.MANUAL_RECOVERY_REQUIRED,
    ]
    recovery_events = tuple(
        event
        for event in loaded.snapshot.events
        if event.type == LEGACY_RECOVERY_EVENT_TYPE
    )
    assert [event.task_id for event in recovery_events] == [
        "task-pending",
        "task-running",
    ]

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT position, agent_sequence, task_id "
            "FROM runtime_events WHERE operation_id = ? AND type = ? "
            "ORDER BY position",
            (legacy.operation.id, LEGACY_RECOVERY_EVENT_TYPE),
        ).fetchall() == [
            (2, 3, "task-pending"),
            (3, 4, "task-running"),
        ]
    finally:
        connection.close()


async def test_version_three_recovery_allocates_across_shared_agent_operations(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy-shared-agent-v3.db"
    primary = await _seed_version_three(path)
    encoded_now = NOW.isoformat(timespec="microseconds").replace("+00:00", "Z")
    secondary_operation_id = "operation-secondary"
    secondary_task_id = "task-secondary-running"
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "INSERT INTO triggers("
            "id, agent_id, kind, source_id, payload_json, created_at, session_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "trigger-secondary",
                primary.operation.agent_id,
                TriggerKind.INTERNAL.value,
                "source-secondary",
                "{}",
                encoded_now,
                "session-secondary",
            ),
        )
        connection.execute(
            "INSERT INTO operations("
            "id, revision, agent_id, trigger_id, status, created_at, updated_at, "
            "session_id, final_text, terminal_reason"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                secondary_operation_id,
                4,
                primary.operation.agent_id,
                "trigger-secondary",
                OperationStatus.RUNNING.value,
                encoded_now,
                encoded_now,
                "session-secondary",
                None,
                None,
            ),
        )
        connection.execute(
            "INSERT INTO turns("
            "operation_id, position, id, number, created_at, model_request_id, "
            "model_response_id"
            ") VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                secondary_operation_id,
                0,
                "turn-secondary",
                1,
                encoded_now,
                "model-call-secondary",
                "model-call-secondary",
            ),
        )
        connection.execute(
            "INSERT INTO model_calls("
            "operation_id, position, id, turn_id, provider_id, request_json, "
            "status, created_at, updated_at, response_json, error_code, "
            "cancellation_requested"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                secondary_operation_id,
                0,
                "model-call-secondary",
                "turn-secondary",
                "legacy:secondary",
                "{}",
                ModelCallStatus.COMPLETED.value,
                encoded_now,
                encoded_now,
                None,
                None,
                0,
            ),
        )
        connection.execute(
            "INSERT INTO tasks("
            "operation_id, position, id, turn_id, call_id, capability_id, "
            "executor_id, status, attempt, arguments_json, created_at, "
            "updated_at, error_code, cancellation_requested"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                secondary_operation_id,
                0,
                secondary_task_id,
                "turn-secondary",
                "call-secondary",
                "legacy.read",
                "legacy.executor",
                TaskStatus.RUNNING.value,
                1,
                '{"key":"secondary"}',
                encoded_now,
                encoded_now,
                None,
                0,
            ),
        )
        connection.execute(
            "INSERT INTO runtime_events("
            "id, operation_id, position, type, agent_id, agent_sequence, "
            "created_at, session_id, payload_json"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "event-secondary-created",
                secondary_operation_id,
                0,
                "operation.created",
                primary.operation.agent_id,
                2,
                encoded_now,
                "session-secondary",
                "{}",
            ),
        )
        connection.commit()
    finally:
        connection.close()

    upgraded = await SQLiteOperationStore.open(path)
    await upgraded.close()

    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT id, revision FROM operations " "WHERE id IN (?, ?) ORDER BY id",
            (primary.operation.id, secondary_operation_id),
        ).fetchall() == [
            (primary.operation.id, 2),
            (secondary_operation_id, 5),
        ]
        assert connection.execute(
            "SELECT operation_id, position, agent_sequence, task_id "
            "FROM runtime_events WHERE agent_id = ? AND type = ? "
            "ORDER BY agent_sequence",
            (primary.operation.agent_id, LEGACY_RECOVERY_EVENT_TYPE),
        ).fetchall() == [
            (primary.operation.id, 1, 3, "task-running"),
            (secondary_operation_id, 1, 4, secondary_task_id),
        ]
    finally:
        connection.close()


async def test_failed_migration_four_restores_the_exact_version_three_image(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed-v4.db"
    backup_path = tmp_path / "failed-v4.exact-backup.db"
    legacy = await _seed_version_three(path)
    version_three_image = _logical_database_image(path)
    migration_four = _migration_four()
    failing_migration = sqlite_owner._SQLiteMigration(
        version=4,
        name=migration_four.name,
        statements=(
            *migration_four.statements,
            "INSERT INTO migration_four_forced_failure(value) VALUES (1)",
        ),
    )

    with pytest.raises(
        SQLiteMigrationError,
        match=rf"migration 4 \({MIGRATION_FOUR_NAME}\) failed",
    ):
        await sqlite_owner._open_with_migrations(
            path,
            migrations=(*_migration_prefix(), failing_migration),
            backup_path=backup_path,
            verify_owned_schema=True,
        )

    assert _logical_database_image(path) == version_three_image
    assert _logical_database_image(backup_path) == version_three_image
    assert _migration_rows(path) == tuple(
        (migration.version, migration.name, migration.checksum)
        for migration in _migration_prefix()
    )
    assert "task_dependencies" not in _schema_names(path, "table")
    assert "task_leases" not in _schema_names(path, "table")
    assert not set(TASK_EXECUTION_COLUMNS).intersection(_columns(path, "tasks"))
    assert _schema_names(path, "trigger").isdisjoint(TASK_EXECUTION_TRIGGERS)

    reopened = await sqlite_owner._open_with_migrations(
        path,
        migrations=_migration_prefix(),
        verify_owned_schema=True,
    )
    await reopened.close()
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT status FROM tasks WHERE operation_id = ? ORDER BY position",
            (legacy.operation.id,),
        ).fetchall() == [
            (TaskStatus.PENDING.value,),
            (TaskStatus.RUNNING.value,),
        ]
    finally:
        connection.close()


async def test_failed_migration_five_restores_the_exact_version_four_image(
    tmp_path: Path,
) -> None:
    path = tmp_path / "failed-v5.db"
    backup_path = tmp_path / "failed-v5.exact-backup.db"
    legacy, evidence_id = await _seed_version_four_with_evidence(path)
    version_four_image = _logical_database_image(path)
    version_four_migrations = (*_migration_prefix(), _migration_four())
    migration_five = _migration_five()
    failing_migration = sqlite_owner._SQLiteMigration(
        version=5,
        name=migration_five.name,
        statements=(
            *migration_five.statements,
            "INSERT INTO migration_five_forced_failure(value) VALUES (1)",
        ),
    )

    with pytest.raises(
        SQLiteMigrationError,
        match=rf"migration 5 \({MIGRATION_FIVE_NAME}\) failed",
    ):
        await sqlite_owner._open_with_migrations(
            path,
            migrations=(*version_four_migrations, failing_migration),
            backup_path=backup_path,
            verify_owned_schema=True,
        )

    assert _logical_database_image(path) == version_four_image
    assert _logical_database_image(backup_path) == version_four_image
    assert _migration_rows(path) == tuple(
        (migration.version, migration.name, migration.checksum)
        for migration in version_four_migrations
    )
    assert "blob_id" not in _columns(path, "evidence")
    connection = sqlite3.connect(path)
    try:
        assert connection.execute(
            "SELECT id FROM evidence WHERE operation_id = ? AND id = ?",
            (legacy.operation.id, evidence_id),
        ).fetchone() == (evidence_id,)
    finally:
        connection.close()

    reopened = await sqlite_owner._open_with_migrations(
        path,
        migrations=version_four_migrations,
        verify_owned_schema=True,
    )
    await reopened.close()
