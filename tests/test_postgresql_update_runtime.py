from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from daita._json import FrozenJsonObject
from daita.adapters import (
    postgresql as postgresql_module,
    postgresql_write as write_module,
)
from daita.adapters.models import SourceRegistration, source_registration_id
from daita.capabilities import (
    AccessMode,
    ApprovalDecision,
    ApprovalRequest,
    Capability,
    CapabilityRegistry,
    ToolExecution,
    ToolOutput,
)
from daita.catalog.models import ResourceKind, TabularColumn
from daita.domains.data.controller import (
    POSTGRESQL_UPDATE_CAPABILITY_ID,
    DataToolRuntime,
)
from daita.domains.data.sql import (
    PostgreSQLUpdateCommand,
    PostgreSQLUpdateIntent,
    ResourceSchema,
)
from daita.llm.models import ToolCall
from daita.loop.models import RunInput
from daita.security import EmptySecretProvider
from daita.storage.sqlite import DatabaseWriteOutcome, SQLiteStateStore

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
SOURCE_ID = source_registration_id(
    "agent-update", "postgresql", "postgresql:update-contract"
)
RESOURCE_ID = "catalog-resource:sha256:" + "2" * 64
SOURCE_REVISION = "catalog:sha256:" + "3" * 64
RESOURCE_REVISION = "sha256:" + "4" * 64


class _SourceStore:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

    async def register_source(self, registration):
        self.registration = registration
        return registration

    async def load_source(self, agent_id: str, source_id: str):
        if (agent_id, source_id) == (
            self.registration.agent_id,
            self.registration.id,
        ):
            return self.registration
        return None

    async def list_sources(self, agent_id: str):
        return (self.registration,) if agent_id == self.registration.agent_id else ()

    async def detach_source(self, agent_id, source_id, detached_at):
        self.registration = self.registration.detach(detached_at)
        return self.registration


class _Catalog:
    async def resource_schemas(self, agent_id: str, source_id: str):
        del agent_id
        return (_resource(),) if source_id == SOURCE_ID else ()

    async def postgresql_update_scope_issue(
        self, agent_id, source_id, resource_id, assignment_columns
    ):
        del agent_id, source_id, resource_id, assignment_columns
        return None


class _Transaction:
    def __init__(
        self,
        log: list[tuple[object, ...]],
        *,
        commit_error: BaseException | None = None,
    ) -> None:
        self.log = log
        self.commit_error = commit_error

    async def start(self) -> None:
        self.log.append(("transaction.start",))

    async def commit(self) -> None:
        self.log.append(("transaction.commit",))
        if self.commit_error is not None:
            raise self.commit_error

    async def rollback(self) -> None:
        self.log.append(("transaction.rollback",))


class _Cursor:
    def __init__(self, rows: tuple[Mapping[str, object], ...]) -> None:
        self._iterator = iter(rows)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration from None


class _Connection:
    def __init__(
        self,
        rows: tuple[Mapping[str, object], ...],
        *,
        update_status: str | BaseException = "UPDATE 3",
        commit_error: BaseException | None = None,
    ) -> None:
        self.rows = rows
        self.update_status = update_status
        self.log: list[tuple[object, ...]] = []
        self.transaction_record = _Transaction(self.log, commit_error=commit_error)

    def transaction(self, **kwargs: object):
        self.log.append(("transaction", kwargs))
        return self.transaction_record

    async def execute(self, sql: str, *parameters: object, **kwargs: object):
        self.log.append(("execute", sql, parameters, kwargs))
        if sql.startswith("UPDATE"):
            if isinstance(self.update_status, BaseException):
                raise self.update_status
            return self.update_status
        return "SELECT 1"

    async def fetchrow(self, sql: str, *parameters: object, **kwargs: object):
        self.log.append(("fetchrow", sql, parameters, kwargs))
        return _guardrails()

    async def fetch(self, sql: str, *parameters: object, **kwargs: object):
        self.log.append(("fetch", sql, parameters, kwargs))
        assert sql.startswith("EXPLAIN")
        return ({"QUERY PLAN": ()},)

    def cursor(self, sql: str, *parameters: object):
        self.log.append(("cursor", sql, parameters))
        return _Cursor(self.rows)

    async def close(self) -> None:
        self.log.append(("close",))

    def terminate(self) -> None:
        self.log.append(("terminate",))


def _registration() -> SourceRegistration:
    return SourceRegistration.build(
        agent_id="agent-update",
        adapter_id="postgresql",
        native_identity="postgresql:update-contract",
        display_name="Update PostgreSQL",
        configuration={
            "database": "warehouse",
            "host": "db.example.test",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "writer",
        },
        attached_at=NOW,
    )


def _resource() -> ResourceSchema:
    return ResourceSchema(
        resource_id=RESOURCE_ID,
        source_id=SOURCE_ID,
        name="accounts",
        aliases=("public.accounts",),
        columns=("account_id", "status", "priority"),
        revision=RESOURCE_REVISION,
        source_revision=SOURCE_REVISION,
        resource_kind="table",
        writable=True,
        primary_key_columns=("account_id",),
        column_nullability=(
            ("account_id", False),
            ("status", False),
            ("priority", False),
        ),
        column_type_provenance=(
            ("account_id", "pg_catalog", "int8"),
            ("status", "pg_catalog", "text"),
            ("priority", "pg_catalog", "int4"),
        ),
        updatable_columns=("status", "priority"),
    )


def _intent() -> PostgreSQLUpdateIntent:
    return PostgreSQLUpdateIntent.from_mapping(
        {
            "source_id": SOURCE_ID,
            "resource_id": RESOURCE_ID,
            "where": [
                {"column": "status", "operator": "eq", "value": "active"},
                {"column": "priority", "operator": "lte", "value": 2},
            ],
            "assignments": [{"column": "status", "value": "inactive"}],
        }
    )


def _priority_intent() -> PostgreSQLUpdateIntent:
    return PostgreSQLUpdateIntent.from_mapping(
        {
            "source_id": SOURCE_ID,
            "resource_id": RESOURCE_ID,
            "where": [
                {"column": "status", "operator": "eq", "value": "active"},
            ],
            "assignments": [{"column": "priority", "value": 4}],
        }
    )


def _row(key: int, *, before: object = "active", xmin: str | None = None):
    return {
        "__daita_primary_key_0": key,
        "__daita_before_0": before,
        "__daita_within_preview_limit": True,
        "__daita_xmin": xmin or str(700 + key),
    }


def _guardrails() -> dict[str, object]:
    return {
        "relation_oid": "16384",
        "relation_kind": "r",
        "is_partition": False,
        "row_level_security": False,
        "force_row_level_security": False,
        "has_inheritance": False,
        "has_user_triggers": False,
        "has_rewrite_rules": False,
        "role_superuser": False,
        "role_bypass_rls": False,
        "role_create_database": False,
        "role_create_role": False,
        "role_replication": False,
        "can_connect": True,
        "can_use_schema": True,
        "can_select_table": True,
        "can_update_columns": True,
    }


def _structure():
    return postgresql_module.PostgreSQLStructure(
        tables=(
            postgresql_module._TableStructure(
                schema="public",
                name="accounts",
                kind=ResourceKind.TABLE,
                columns=(
                    TabularColumn(
                        name="account_id",
                        native_type="bigint",
                        native_type_namespace="pg_catalog",
                        native_type_name="int8",
                        ordinal=0,
                        nullable=False,
                        primary_key_ordinal=1,
                        updatable=True,
                    ),
                    TabularColumn(
                        name="status",
                        native_type="text",
                        native_type_namespace="pg_catalog",
                        native_type_name="text",
                        ordinal=1,
                        nullable=False,
                        updatable=True,
                    ),
                    TabularColumn(
                        name="priority",
                        native_type="integer",
                        native_type_namespace="pg_catalog",
                        native_type_name="int4",
                        ordinal=2,
                        nullable=False,
                        updatable=True,
                    ),
                ),
                indexes=(),
            ),
        ),
        relationships=(),
        source_revision=SOURCE_REVISION,
    )


def _execution() -> ToolExecution:
    return ToolExecution(
        run_id="run-update",
        call_id="call-update",
        capability_id=POSTGRESQL_UPDATE_CAPABILITY_ID,
    )


def _patch_io(monkeypatch, connections: list[_Connection]) -> None:
    async def connect(*args, **kwargs):
        del args, kwargs
        if not connections:
            raise AssertionError("unexpected reconnect or retry")
        return connections.pop(0)

    async def load_structure(*args, **kwargs):
        del args, kwargs
        return _structure()

    monkeypatch.setattr(write_module, "_connect", connect)
    monkeypatch.setattr(write_module, "_load_structure", load_structure)


def _backend(store: SQLiteStateStore):
    return write_module.PostgreSQLUpdatePreviewBackend(
        _SourceStore(_registration()),
        _Catalog(),
        EmptySecretProvider(),
        receipt_store=store,
        clock=lambda: NOW,
    )


async def _command(
    backend,
    *,
    expected_rows: int = 3,
    intent: PostgreSQLUpdateIntent | None = None,
):
    selected = intent or _intent()
    preview = await backend.preview_update(agent_id="agent-update", intent=selected)
    assert preview.matched_rows == expected_rows
    return PostgreSQLUpdateCommand(
        intent=selected,
        preview_fingerprint=preview.fingerprint.preview_fingerprint,
        expected_affected_rows=preview.matched_rows,
    )


async def test_bulk_update_commits_once_with_exact_receipt(monkeypatch, tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    preview_connection = _Connection(tuple(_row(index) for index in (1, 2, 3)))
    write_connection = _Connection(tuple(_row(index) for index in (1, 2, 3)))
    _patch_io(monkeypatch, [preview_connection, write_connection])
    backend = _backend(store)
    try:
        command = await _command(backend)
        result = await backend.execute_update(
            agent_id="agent-update", execution=_execution(), command=command
        )
        assert result.affected_rows == 3
        assert result.target_set_sha256.startswith("sha256:")
        update_calls = [
            item
            for item in write_connection.log
            if item[0] == "execute" and str(item[1]).startswith("UPDATE")
        ]
        assert len(update_calls) == 1
        assert update_calls[0][2] == ("inactive", "active", 2)
        locked = next(item for item in write_connection.log if item[0] == "cursor")
        assert str(locked[1]).endswith(" FOR UPDATE")
        receipt = await store.load_database_write_receipt(
            "agent-update", result.receipt_id
        )
        assert receipt is not None
        assert receipt.outcome is DatabaseWriteOutcome.COMMITTED
        assert receipt.expected_affected_rows == 3
        assert receipt.affected_rows == 3
    finally:
        await store.close()


async def test_target_set_drift_rolls_back_without_update(monkeypatch, tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    preview_connection = _Connection(tuple(_row(index) for index in (1, 2, 3)))
    write_connection = _Connection(tuple(_row(index) for index in (1, 2)))
    _patch_io(monkeypatch, [preview_connection, write_connection])
    backend = _backend(store)
    try:
        command = await _command(backend)
        with pytest.raises(write_module.PostgreSQLUpdateExecutionError) as captured:
            await backend.execute_update(
                agent_id="agent-update", execution=_execution(), command=command
            )
        assert captured.value.error_code == "write_state_changed"
        assert not any(
            item[0] == "execute" and str(item[1]).startswith("UPDATE")
            for item in write_connection.log
        )
        receipt = await store.load_database_write_receipt_for_call(
            "agent-update", "run-update", "call-update"
        )
        assert receipt is not None
        assert receipt.outcome is DatabaseWriteOutcome.NOT_COMMITTED
    finally:
        await store.close()


async def test_assigned_value_drift_rolls_back_without_update(monkeypatch, tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    intent = _priority_intent()
    preview_connection = _Connection(
        tuple(_row(index, before=1) for index in (1, 2, 3))
    )
    write_connection = _Connection(
        (
            _row(1, before=1),
            _row(2, before=2),
            _row(3, before=1),
        )
    )
    _patch_io(monkeypatch, [preview_connection, write_connection])
    backend = _backend(store)
    try:
        command = await _command(backend, intent=intent)
        with pytest.raises(write_module.PostgreSQLUpdateExecutionError) as captured:
            await backend.execute_update(
                agent_id="agent-update", execution=_execution(), command=command
            )
        assert captured.value.error_code == "write_state_changed"
        assert not any(
            item[0] == "execute" and str(item[1]).startswith("UPDATE")
            for item in write_connection.log
        )
    finally:
        await store.close()


class _RuntimeCatalog:
    async def source_routing_facts(self, agent_id: str, source_ids=()):
        del agent_id, source_ids
        return ()


class _AtomicUpdateExecutor:
    executor_id = "test.postgresql.atomic_update"

    def __init__(self) -> None:
        self.preflight_count = 0
        self.execute_count = 0

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        del request
        self.preflight_count += 1
        return FrozenJsonObject.from_mapping({"fingerprint": "current"})

    async def execute(self, request: ToolExecution) -> ToolOutput:
        del request
        self.execute_count += 1
        return ToolOutput(kind="test.postgresql.update", data={"committed": True})


async def test_runtime_omits_only_redundant_post_approval_update_preflight():
    executor = _AtomicUpdateExecutor()
    capability = Capability(
        id=POSTGRESQL_UPDATE_CAPABILITY_ID,
        description="test atomic PostgreSQL update",
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        output_kind="test.postgresql.update",
        output_schema={
            "type": "object",
            "properties": {"committed": {"type": "boolean"}},
            "required": ["committed"],
            "additionalProperties": False,
        },
        executor_id=executor.executor_id,
        access_mode=AccessMode.WRITE,
        side_effecting=True,
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    runtime = DataToolRuntime(
        CapabilityRegistry(capabilities=(capability,), executors=(executor,)),
        _RuntimeCatalog(),  # type: ignore[arg-type]
        approval_handler=approve,
    )
    run = RunInput(
        id="run-runtime-update",
        agent_id="agent-update",
        message="update",
        created_at=NOW,
        conversation_id="conversation-runtime-update",
    )
    call = ToolCall(id="call-runtime-update", name="test_update", arguments={})
    execution = ToolExecution(
        run_id=run.id,
        call_id=call.id,
        capability_id=capability.id,
        arguments={},
        conversation_id=run.conversation_id,
    )

    result, interruption, certainty = await runtime._execute_side_effect(
        run,
        call,
        capability,
        executor,
        execution,
    )

    assert not result.is_error
    assert interruption is None
    assert certainty.value == "definite"
    assert len(approvals) == 1
    assert executor.preflight_count == 1
    assert executor.execute_count == 1


async def test_affected_row_mismatch_rolls_back(monkeypatch, tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    rows = tuple(_row(index) for index in (1, 2, 3))
    preview_connection = _Connection(rows)
    write_connection = _Connection(rows, update_status="UPDATE 2")
    _patch_io(monkeypatch, [preview_connection, write_connection])
    backend = _backend(store)
    try:
        command = await _command(backend)
        with pytest.raises(write_module.PostgreSQLUpdateExecutionError) as captured:
            await backend.execute_update(
                agent_id="agent-update", execution=_execution(), command=command
            )
        assert captured.value.error_code == "write_affected_rows_mismatch"
        assert ("transaction.rollback",) in write_connection.log
    finally:
        await store.close()


async def test_commit_uncertainty_is_recorded_and_never_retried(monkeypatch, tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    rows = tuple(_row(index) for index in (1, 2, 3))
    preview_connection = _Connection(rows)
    write_connection = _Connection(rows, commit_error=ConnectionError("lost"))
    _patch_io(monkeypatch, [preview_connection, write_connection])
    backend = _backend(store)
    try:
        command = await _command(backend)
        with pytest.raises(write_module.PostgreSQLUpdateExecutionError) as captured:
            await backend.execute_update(
                agent_id="agent-update", execution=_execution(), command=command
            )
        assert captured.value.error_code == "write_outcome_unknown"
        receipt = await store.load_database_write_receipt_for_call(
            "agent-update", "run-update", "call-update"
        )
        assert receipt is not None
        assert receipt.outcome is DatabaseWriteOutcome.OUTCOME_UNKNOWN
        assert (
            len(
                [
                    item
                    for item in write_connection.log
                    if item[0] == "execute" and str(item[1]).startswith("UPDATE")
                ]
            )
            == 1
        )
    finally:
        await store.close()


async def test_duplicate_execution_identity_never_reconnects(monkeypatch, tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    rows = tuple(_row(index) for index in (1, 2, 3))
    preview_connection = _Connection(rows)
    write_connection = _Connection(rows)
    connections = [preview_connection, write_connection]
    _patch_io(monkeypatch, connections)
    backend = _backend(store)
    try:
        command = await _command(backend)
        await backend.execute_update(
            agent_id="agent-update", execution=_execution(), command=command
        )
        with pytest.raises(write_module.PostgreSQLUpdateExecutionError) as captured:
            await backend.execute_update(
                agent_id="agent-update", execution=_execution(), command=command
            )
        assert captured.value.error_code == "write_execution_duplicate"
        assert connections == []
    finally:
        await store.close()
