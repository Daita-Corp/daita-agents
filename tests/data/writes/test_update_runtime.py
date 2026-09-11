from __future__ import annotations

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
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    EffectReceiptPolicy,
    OperationalEffect,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolView,
)
from daita.capability_runtime import CapabilityRuntime
from daita.catalog.models import ResourceKind, TabularColumn
from daita.domains.data.controller import (
    RELATIONAL_UPDATE_CAPABILITY_ID,
)
from daita.domains.data.sql import (
    RelationalUpdateCommand,
    RelationalUpdateIntent,
    ResourceSchema,
)
from daita.identity import AgentIdentity
from daita.llm.models import ToolCall
from daita.loop.models import RunInput
from daita.security import EmptySecretProvider
from daita.storage.sqlite import SQLiteStateStore
from daita.storage.sqlite_records import RelationalWriteScope
from tests.data.writes._update_runtime_support import _Connection, _row
from tests.support.capability_runtime import (
    StaticTestDomain,
    execute_projected,
    presentation_metadata,
    static_registry,
)

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

    async def load_relational_write_scope(self, agent_id, source_id, resource_id):
        return RelationalWriteScope(
            agent_id=agent_id,
            source_id=source_id,
            resource_id=resource_id,
            resource_revision=RESOURCE_REVISION,
            allowed_operations=("update",),
            allowed_insert_columns=(),
            allowed_update_columns=("status", "priority"),
            key_columns=("account_id",),
            generated_identity_columns=(),
            max_rows=10000,
            authorization_fingerprint="sha256:" + "9" * 64,
        )

    async def relational_write_scope_issue(
        self, agent_id, source_id, resource_id, assignment_columns, **kwargs
    ):
        del agent_id, source_id, resource_id, assignment_columns
        return None


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


def _intent() -> RelationalUpdateIntent:
    return RelationalUpdateIntent.from_mapping(
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


def _priority_intent() -> RelationalUpdateIntent:
    return RelationalUpdateIntent.from_mapping(
        {
            "source_id": SOURCE_ID,
            "resource_id": RESOURCE_ID,
            "where": [
                {"column": "status", "operator": "eq", "value": "active"},
            ],
            "assignments": [{"column": "priority", "value": 4}],
        }
    )


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
        capability_id=RELATIONAL_UPDATE_CAPABILITY_ID,
        effect_receipt_id="effect-receipt:sha256:" + "e" * 64,
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
    return write_module.PostgreSQLWriteBackend(
        _SourceStore(_registration()),
        _Catalog(),
        EmptySecretProvider(),
        clock=lambda: NOW,
    )


async def _command(
    backend,
    *,
    expected_rows: int = 3,
    intent: RelationalUpdateIntent | None = None,
):
    selected = intent or _intent()
    preview = await backend.preview_update(agent_id="agent-update", intent=selected)
    assert preview.matched_rows == expected_rows
    return RelationalUpdateCommand(
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
        observation = result.effect_observation
        assert observation.outcome is EffectOutcome.SUCCEEDED
        assert observation.evidence_basis is EffectEvidenceBasis.ADAPTER_VERIFIED
        assert observation.payload is not None
        assert observation.payload["expected_affected_rows"] == 3
        assert observation.payload["affected_rows"] == 3
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
        with pytest.raises(write_module.RelationalUpdateExecutionError) as captured:
            await backend.execute_update(
                agent_id="agent-update", execution=_execution(), command=command
            )
        assert captured.value.error_code == "write_state_changed"
        assert not any(
            item[0] == "execute" and str(item[1]).startswith("UPDATE")
            for item in write_connection.log
        )
        observation = captured.value.effect_observation
        assert (
            observation is not None and observation.outcome is EffectOutcome.NOT_APPLIED
        )
        assert (
            observation.payload is not None
            and observation.payload["affected_rows"] == 0
        )
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
        with pytest.raises(write_module.RelationalUpdateExecutionError) as captured:
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
        return ToolOutput(
            kind="test.postgresql.update",
            data={"committed": True},
            effect_observation=EffectObservation(
                EffectOutcome.SUCCEEDED,
                EffectEvidenceBasis.ADAPTER_VERIFIED,
                FrozenJsonObject.from_mapping({"committed": True}),
            ),
        )


async def test_runtime_omits_only_redundant_post_approval_update_preflight(tmp_path):
    executor = _AtomicUpdateExecutor()
    capability = Capability(
        id=RELATIONAL_UPDATE_CAPABILITY_ID,
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
        operational_effect=OperationalEffect.MUTATE_DATA,
        effect_receipt_policy=EffectReceiptPolicy(
            "test.atomic",
            {
                "type": "object",
                "properties": {"committed": {"type": "boolean"}},
                "required": ["committed"],
                "additionalProperties": False,
            },
            EffectEvidenceBasis.ADAPTER_VERIFIED,
        ),
    )
    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    view = ToolView(
        name="test_update",
        capability_id=capability.id,
        description=capability.description,
        presentation=presentation_metadata(load_mode=ToolLoadMode.ON_DEMAND),
    )
    domain = StaticTestDomain(
        (capability,),
        (view,),
        recheck_after_approval=False,
    )
    store = await SQLiteStateStore.open(tmp_path / "atomic.db", clock=lambda: NOW)
    await store.initialize_identity(AgentIdentity("agent-update", "Update", NOW))
    runtime = CapabilityRuntime(
        static_registry(domain, (executor,)),
        (domain,),
        approval_handler=approve,
        effect_receipts=store,
        clock=lambda: NOW,
    )
    run = RunInput(
        id="run-runtime-update",
        agent_id="agent-update",
        message="update",
        created_at=NOW,
        conversation_id="conversation-runtime-update",
    )
    call = ToolCall(id="call-runtime-update", name="test_update", arguments={})
    await store.start(run)
    outcome = await execute_projected(
        runtime,
        run,
        (call,),
    )
    (result,) = outcome.ordered_results

    assert not result.is_error
    assert outcome.interruption_kind is None
    assert outcome.outcome_certainty.value == "definite"
    assert len(approvals) == 1
    assert executor.preflight_count == 1
    assert executor.execute_count == 1

    await store.close()


async def test_affected_row_mismatch_rolls_back(monkeypatch, tmp_path):
    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    rows = tuple(_row(index) for index in (1, 2, 3))
    preview_connection = _Connection(rows)
    write_connection = _Connection(rows, update_status="UPDATE 2")
    _patch_io(monkeypatch, [preview_connection, write_connection])
    backend = _backend(store)
    try:
        command = await _command(backend)
        with pytest.raises(write_module.RelationalUpdateExecutionError) as captured:
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
        with pytest.raises(write_module.RelationalUpdateExecutionError) as captured:
            await backend.execute_update(
                agent_id="agent-update", execution=_execution(), command=command
            )
        assert captured.value.error_code == "write_outcome_unknown"
        observation = captured.value.effect_observation
        assert (
            observation is not None and observation.outcome is EffectOutcome.UNCERTAIN
        )
        assert (
            observation.payload is not None
            and observation.payload["affected_rows"] is None
        )
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
    from daita.domains.data.capabilities import RELATIONAL_UPDATE_RECEIPT_POLICY

    store = await SQLiteStateStore.open(tmp_path / "state.db", clock=lambda: NOW)
    await store.initialize_identity(AgentIdentity("agent-update", "Update", NOW))
    rows = tuple(_row(index) for index in (1, 2, 3))
    connections = [_Connection(rows), _Connection(rows)]
    _patch_io(monkeypatch, connections)
    backend = _backend(store)
    command = await _command(backend)

    class NativeExecutor(_AtomicUpdateExecutor):
        async def execute(self, request: ToolExecution) -> ToolOutput:
            self.execute_count += 1
            result = await backend.execute_update(
                agent_id="agent-update", execution=request, command=command
            )
            return ToolOutput(
                kind="test.postgresql.update",
                data={"committed": True},
                effect_observation=result.effect_observation,
            )

    executor = NativeExecutor()
    capability = Capability(
        id=RELATIONAL_UPDATE_CAPABILITY_ID,
        description="Exact native update",
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
        operational_effect=OperationalEffect.MUTATE_DATA,
        effect_receipt_policy=RELATIONAL_UPDATE_RECEIPT_POLICY,
    )
    view = ToolView(
        name="test_update",
        capability_id=capability.id,
        description=capability.description,
        presentation=presentation_metadata(load_mode=ToolLoadMode.ON_DEMAND),
    )
    domain = StaticTestDomain((capability,), (view,), recheck_after_approval=False)

    async def approve(request):
        return ApprovalDecision.APPROVE

    runtime = CapabilityRuntime(
        static_registry(domain, (executor,)),
        (domain,),
        effect_receipts=store,
        approval_handler=approve,
        clock=lambda: NOW,
    )
    run = RunInput(
        id="run-update",
        agent_id="agent-update",
        message="Update",
        created_at=NOW,
        conversation_id="conversation-update",
    )
    await store.start(run)
    call = ToolCall(id="call-update", name="test_update", arguments={})
    try:
        first = (await execute_projected(runtime, run, (call,))).ordered_results[0]
        assert not first.is_error
        second = (await execute_projected(runtime, run, (call,))).ordered_results[0]
        assert (
            second.is_error
            and second.output["error"]["code"] == "effect_already_reserved"
        )
        assert second.output["effect_receipt"] == first.output["effect_receipt"]
        assert executor.execute_count == 1 and connections == []
        receipt = await store.load_effect_receipt_for_call(
            run.agent_id, run.id, call.id
        )
        assert receipt is not None and receipt.outcome is EffectOutcome.SUCCEEDED
        assert receipt.evidence_basis is EffectEvidenceBasis.ADAPTER_VERIFIED
    finally:
        await store.close()
