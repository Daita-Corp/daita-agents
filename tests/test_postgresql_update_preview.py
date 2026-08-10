from __future__ import annotations

import asyncio
from collections.abc import Mapping
from datetime import datetime, timezone

import pytest

from daita import Agent
from daita._json import canonical_json
from daita.adapters import postgresql as postgresql_module
from daita.adapters import postgresql_write as write_module
from daita.adapters.models import (
    DiscoveryRequest,
    SourceRegistration,
    source_registration_id,
)
from daita.catalog.models import ResourceKind, TabularColumn, catalog_resource_id
from daita.capabilities import CapabilityInputError, CapabilityRegistry
from daita.domains.data.capabilities import (
    POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
    postgresql_update_preview_declarations,
)
from daita.domains.data import context as context_module
from daita.domains.data.controller import POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND
from daita.domains.data.sql import PostgreSQLUpdateIntent, ResourceSchema
from daita.llm.models import (
    FinishReason,
    MessageRole,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.security import EmptySecretProvider

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
_NATIVE_IDENTITY = "postgresql:preview-contract"
SOURCE_ID = source_registration_id(
    "agent-preview",
    "postgresql",
    _NATIVE_IDENTITY,
)
RESOURCE_ID = "catalog-resource:sha256:" + "2" * 64
SOURCE_REVISION = "catalog:sha256:" + "3" * 64
RESOURCE_REVISION = "sha256:" + "4" * 64


class _SourceStore:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

    async def register_source(
        self,
        registration: SourceRegistration,
    ) -> SourceRegistration:
        self.registration = registration
        return registration

    async def load_source(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceRegistration | None:
        if self.registration.agent_id == agent_id and self.registration.id == source_id:
            return self.registration
        return None

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        return (self.registration,) if self.registration.agent_id == agent_id else ()

    async def detach_source(
        self,
        agent_id: str,
        source_id: str,
        detached_at: datetime,
    ) -> SourceRegistration:
        assert self.registration.agent_id == agent_id
        assert self.registration.id == source_id
        self.registration = self.registration.detach(detached_at)
        return self.registration


class _Catalog:
    def __init__(self, resource: ResourceSchema) -> None:
        self.resource = resource

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]:
        del agent_id
        return (self.resource,) if source_id == self.resource.source_id else ()


class _Transaction:
    def __init__(self, log: list[tuple[object, ...]]) -> None:
        self.log = log

    async def start(self) -> None:
        self.log.append(("transaction.start",))

    async def commit(self) -> None:
        self.log.append(("transaction.commit",))

    async def rollback(self) -> None:
        self.log.append(("transaction.rollback",))


class _Connection:
    def __init__(
        self,
        *,
        preview_rows: tuple[Mapping[str, object], ...] | BaseException | None = None,
        guardrails: Mapping[str, object] | None = None,
        compile_error: BaseException | None = None,
    ) -> None:
        self.log: list[tuple[object, ...]] = []
        self.transaction_record = _Transaction(self.log)
        self.preview_rows = (_preview_row(),) if preview_rows is None else preview_rows
        self.guardrails = dict(guardrails or _guardrails())
        self.compile_error = compile_error
        self.terminated = False

    def transaction(self, **kwargs: object) -> _Transaction:
        self.log.append(("transaction", kwargs))
        return self.transaction_record

    async def execute(self, sql: str, *parameters: object) -> str:
        self.log.append(("execute", sql, parameters))
        return "SELECT 1"

    async def fetchrow(
        self,
        sql: str,
        *parameters: object,
        **kwargs: object,
    ) -> Mapping[str, object]:
        self.log.append(("fetchrow", sql, parameters, kwargs))
        assert "update_preview_guardrails" in sql
        return self.guardrails

    async def fetch(
        self,
        sql: str,
        *parameters: object,
        **kwargs: object,
    ) -> tuple[Mapping[str, object], ...]:
        self.log.append(("fetch", sql, parameters, kwargs))
        if sql.startswith("EXPLAIN"):
            if self.compile_error is not None:
                raise self.compile_error
            return ({"QUERY PLAN": ()},)
        assert "update_preview_row" in sql
        if isinstance(self.preview_rows, BaseException):
            raise self.preview_rows
        return self.preview_rows

    async def close(self) -> None:
        self.log.append(("close",))

    def terminate(self) -> None:
        self.terminated = True
        self.log.append(("terminate",))


class _SqlStateError(RuntimeError):
    def __init__(self, sqlstate: str, diagnostic: str) -> None:
        super().__init__(diagnostic)
        self.sqlstate = sqlstate


def _registration(
    *,
    agent_id: str = "agent-preview",
    write_access: bool = True,
    adapter_id: str = "postgresql",
) -> SourceRegistration:
    registration = SourceRegistration.build(
        agent_id=agent_id,
        adapter_id=adapter_id,
        native_identity=_NATIVE_IDENTITY,
        display_name="Preview PostgreSQL",
        configuration={
            "database": "warehouse",
            "host": "db.example.test",
            "port": 5432,
            "schemas": ("public",),
            "ssl_mode": "require",
            "username": "writer",
            "write_access": write_access,
        },
        attached_at=NOW,
    )
    return registration


def _resource(**changes: object) -> ResourceSchema:
    values: dict[str, object] = {
        "resource_id": RESOURCE_ID,
        "source_id": SOURCE_ID,
        "name": "accounts",
        "aliases": ("public.accounts",),
        "columns": ("account_id", "status", "metadata"),
        "revision": RESOURCE_REVISION,
        "source_revision": SOURCE_REVISION,
        "resource_kind": "table",
        "writable": True,
        "primary_key_columns": ("account_id",),
        "column_nullability": (
            ("account_id", False),
            ("status", False),
            ("metadata", True),
        ),
        "column_type_provenance": (
            ("account_id", "pg_catalog", "int8"),
            ("status", "pg_catalog", "text"),
            ("metadata", "pg_catalog", "jsonb"),
        ),
        "updatable_columns": ("status", "metadata"),
    }
    values.update(changes)
    return ResourceSchema(**values)  # type: ignore[arg-type]


def _intent(
    *,
    value: object = "inactive",
    source_id: str = SOURCE_ID,
    resource_id: str = RESOURCE_ID,
) -> PostgreSQLUpdateIntent:
    return PostgreSQLUpdateIntent.from_mapping(
        {
            "source_id": source_id,
            "resource_id": resource_id,
            "match": [{"column": "account_id", "value": 42}],
            "assignments": [{"column": "status", "value": value}],
        }
    )


def _guardrails(**changes: object) -> dict[str, object]:
    facts: dict[str, object] = {
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
    facts.update(changes)
    return facts


def _preview_row(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "__daita_primary_key_0": 42,
        "__daita_before_0": "active",
        "__daita_within_preview_limit": True,
        "__daita_tableoid": "16384",
        "__daita_ctid": "(0,1)",
        "__daita_xmin": "751",
    }
    row.update(changes)
    return row


def _structure(*, source_revision: str = SOURCE_REVISION):
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
                        name="metadata",
                        native_type="jsonb",
                        native_type_namespace="pg_catalog",
                        native_type_name="jsonb",
                        ordinal=2,
                        nullable=True,
                        updatable=True,
                    ),
                ),
                indexes=(),
            ),
        ),
        relationships=(),
        source_revision=source_revision,
    )


def _backend(
    connection: _Connection,
    *,
    registration: SourceRegistration | None = None,
    resource: ResourceSchema | None = None,
) -> write_module.PostgreSQLUpdatePreviewBackend:
    return write_module.PostgreSQLUpdatePreviewBackend(
        _SourceStore(registration or _registration()),
        _Catalog(resource or _resource()),
        EmptySecretProvider(),
    )


def _patch_io(
    monkeypatch: pytest.MonkeyPatch,
    connection: _Connection,
    *,
    structure: object | None = None,
) -> None:
    async def connect(*args: object, **kwargs: object) -> _Connection:
        del args, kwargs
        return connection

    async def load_structure(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return structure or _structure()

    monkeypatch.setattr(write_module, "_connect", connect)
    monkeypatch.setattr(write_module, "_load_structure", load_structure)


async def test_preview_is_read_only_parameterized_bounded_and_secret_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = "inactive'; DELETE FROM audit; --"
    connection = _Connection()
    _patch_io(monkeypatch, connection)

    preview = await _backend(connection).preview_update(
        agent_id="agent-preview",
        intent=_intent(value=value),
    )
    data = preview.tool_data().to_dict()

    assert connection.log[0] == (
        "transaction",
        {"isolation": "repeatable_read", "readonly": True},
    )
    assert ("transaction.start",) in connection.log
    assert ("transaction.commit",) in connection.log
    assert ("transaction.rollback",) not in connection.log
    assert connection.log[-1] == ("close",)
    sql_calls = tuple(
        entry
        for entry in connection.log
        if entry[0] in {"execute", "fetch", "fetchrow"}
    )
    rendered_sql = tuple(str(entry[1]) for entry in sql_calls)
    assert not any(sql.lstrip().startswith("UPDATE") for sql in rendered_sql)
    explain = next(sql for sql in rendered_sql if sql.startswith("EXPLAIN"))
    assert "ANALYZE" not in explain
    assert 'UPDATE ONLY "public"."accounts" SET "status" = $1' in explain
    assert value not in explain
    compile_call = next(
        entry
        for entry in connection.log
        if entry[0] == "fetch" and str(entry[1]).startswith("EXPLAIN")
    )
    assert compile_call[2] == (value, 42)
    preview_call = next(
        entry
        for entry in connection.log
        if entry[0] == "fetch" and "update_preview_row" in str(entry[1])
    )
    assert preview_call[2] == (42,)
    assert "LIMIT 2" in str(preview_call[1])
    assert data["would_affect"] == 1
    assert data["before"] == [{"column": "status", "value": "active"}]
    assert data["after"] == [{"column": "status", "value": value}]
    assert data["trust_classification"] == "untrusted_data"
    assert len(canonical_json(data).encode("utf-8")) < 256 * 1_024
    serialized = canonical_json(data)
    for forbidden in (
        "UPDATE ONLY",
        "db.example.test",
        "writer",
        "credential",
        "password",
        "connection",
    ):
        assert forbidden not in serialized


async def test_zero_row_preview_is_successful_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_connection = _Connection(preview_rows=())
    _patch_io(monkeypatch, first_connection)
    first = await _backend(first_connection).preview_update(
        agent_id="agent-preview",
        intent=_intent(),
    )
    second_connection = _Connection(preview_rows=())
    _patch_io(monkeypatch, second_connection)
    second = await _backend(second_connection).preview_update(
        agent_id="agent-preview",
        intent=_intent(),
    )

    assert first.would_affect == 0
    assert first.before == first.after == ()
    assert first.warnings == ("target_not_found",)
    assert first.fingerprint == second.fingerprint
    assert first.fingerprint.intent_sha256 == second.fingerprint.intent_sha256


async def test_jsonb_values_are_bound_and_returned_as_structured_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    intent = PostgreSQLUpdateIntent.from_mapping(
        {
            "source_id": SOURCE_ID,
            "resource_id": RESOURCE_ID,
            "match": [{"column": "account_id", "value": 42}],
            "assignments": [{"column": "metadata", "value": {"reason": "closed"}}],
        }
    )
    connection = _Connection(
        preview_rows=(_preview_row(__daita_before_0='{"reason":"opened"}'),)
    )
    _patch_io(monkeypatch, connection)

    preview = await _backend(connection).preview_update(
        agent_id="agent-preview",
        intent=intent,
    )

    assert preview.tool_data().to_dict()["before"] == [
        {"column": "metadata", "value": {"reason": "opened"}}
    ]
    compile_call = next(
        entry
        for entry in connection.log
        if entry[0] == "fetch" and str(entry[1]).startswith("EXPLAIN")
    )
    assert compile_call[2] == ('{"reason":"closed"}', 42)


async def test_guardrail_and_row_changes_alter_preview_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_connection = _Connection()
    _patch_io(monkeypatch, first_connection)
    first = await _backend(first_connection).preview_update(
        agent_id="agent-preview",
        intent=_intent(),
    )

    relation_connection = _Connection(guardrails=_guardrails(relation_oid="16385"))
    _patch_io(monkeypatch, relation_connection)
    relation = await _backend(relation_connection).preview_update(
        agent_id="agent-preview",
        intent=_intent(),
    )

    row_connection = _Connection(preview_rows=(_preview_row(__daita_xmin="752"),))
    _patch_io(monkeypatch, row_connection)
    row = await _backend(row_connection).preview_update(
        agent_id="agent-preview",
        intent=_intent(),
    )

    assert (
        first.fingerprint.preview_fingerprint
        != relation.fingerprint.preview_fingerprint
    )
    assert first.fingerprint.row_version_sha256 != row.fingerprint.row_version_sha256
    assert first.fingerprint.preview_fingerprint != row.fingerprint.preview_fingerprint


@pytest.mark.parametrize(
    "changed",
    (
        {"relation_kind": "p"},
        {"is_partition": True},
        {"row_level_security": True},
        {"force_row_level_security": True},
        {"has_inheritance": True},
        {"has_user_triggers": True},
        {"has_rewrite_rules": True},
        {"role_superuser": True},
        {"role_bypass_rls": True},
        {"role_create_database": True},
        {"role_create_role": True},
        {"role_replication": True},
        {"can_connect": False},
        {"can_use_schema": False},
        {"can_select_table": False},
        {"can_update_columns": False},
    ),
)
async def test_live_relation_role_and_privilege_guardrails_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    changed: dict[str, object],
) -> None:
    connection = _Connection(guardrails=_guardrails(**changed))
    _patch_io(monkeypatch, connection)

    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as raised:
        await _backend(connection).preview_update(
            agent_id="agent-preview",
            intent=_intent(),
        )

    assert raised.value.code == "write_guardrail_rejected"
    assert ("transaction.rollback",) in connection.log
    assert connection.log[-1] == ("close",)
    assert not any(
        entry[0] == "fetch" and str(entry[1]).startswith("EXPLAIN")
        for entry in connection.log
    )


async def test_stale_live_structure_and_catalog_revision_fail_before_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection()
    _patch_io(
        monkeypatch,
        connection,
        structure=_structure(source_revision="catalog:sha256:" + "9" * 64),
    )
    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as stale_source:
        await _backend(connection).preview_update(
            agent_id="agent-preview",
            intent=_intent(),
        )
    assert stale_source.value.code == "write_resource_not_writable"

    catalog_connection = _Connection()
    connected = False

    async def unexpected_connect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal connected
        connected = True
        raise AssertionError("stale catalog must fail before source I/O")

    monkeypatch.setattr(write_module, "_connect", unexpected_connect)
    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as stale_resource:
        await _backend(
            catalog_connection,
            resource=_resource(revision=None),
        ).preview_update(agent_id="agent-preview", intent=_intent())
    assert stale_resource.value.code == "write_resource_not_writable"
    assert connected is False


async def test_write_access_and_source_ownership_are_rechecked_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connected = False

    async def unexpected_connect(*args: object, **kwargs: object) -> object:
        del args, kwargs
        nonlocal connected
        connected = True
        raise AssertionError("ineligible source must fail before connection")

    monkeypatch.setattr(write_module, "_connect", unexpected_connect)
    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as disabled:
        await _backend(
            _Connection(),
            registration=_registration(write_access=False),
        ).preview_update(agent_id="agent-preview", intent=_intent())
    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as foreign:
        await _backend(
            _Connection(),
            registration=_registration(agent_id="another-agent"),
        ).preview_update(agent_id="agent-preview", intent=_intent())

    assert disabled.value.code == "write_access_not_enabled"
    assert foreign.value.code == "write_source_not_available"
    assert connected is False


async def test_more_than_one_row_and_oversized_before_image_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    two_rows = _Connection(preview_rows=(_preview_row(), _preview_row()))
    _patch_io(monkeypatch, two_rows)
    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as cardinality:
        await _backend(two_rows).preview_update(
            agent_id="agent-preview",
            intent=_intent(),
        )
    assert cardinality.value.code == "write_guardrail_rejected"

    oversized = _Connection(
        preview_rows=(_preview_row(__daita_within_preview_limit=False),)
    )
    _patch_io(monkeypatch, oversized)
    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as bounded:
        await _backend(oversized).preview_update(
            agent_id="agent-preview",
            intent=_intent(),
        )
    assert bounded.value.code == "write_preview_failed"


async def test_cancellation_rolls_back_and_closes_the_read_only_transaction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = _Connection(preview_rows=asyncio.CancelledError())
    _patch_io(monkeypatch, connection)

    with pytest.raises(asyncio.CancelledError):
        await _backend(connection).preview_update(
            agent_id="agent-preview",
            intent=_intent(),
        )

    assert ("transaction.rollback",) in connection.log
    assert connection.log[-1] == ("close",)
    assert not connection.terminated


async def test_compile_errors_are_normalized_without_raw_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostic = "host=db.internal password=database-secret UPDATE accounts"
    connection = _Connection(compile_error=_SqlStateError("42804", diagnostic))
    _patch_io(monkeypatch, connection)

    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as raised:
        await _backend(connection).preview_update(
            agent_id="agent-preview",
            intent=_intent(),
        )

    assert raised.value.code == "write_compile_failed"
    assert diagnostic not in str(raised.value)
    assert "password" not in str(raised.value)
    assert ("transaction.rollback",) in connection.log
    assert connection.log[-1] == ("close",)


def test_preview_schema_rejects_model_owned_execution_identity_fields() -> None:
    connection = _Connection()
    declarations = postgresql_update_preview_declarations(
        "agent-preview",
        _backend(connection),
    )
    registry = CapabilityRegistry(
        capabilities=declarations.capabilities,
        executors=declarations.executors,
        tool_views=declarations.tool_views,
    )
    capability_id = declarations.capabilities[0].id
    arguments = _intent().to_payload()
    for forbidden in (
        "call_id",
        "receipt_id",
        "preview_fingerprint",
        "idempotency_key",
    ):
        with pytest.raises(CapabilityInputError) as raised:
            registry.validate_arguments(
                capability_id,
                {**arguments, forbidden: "model-authored"},
            )
        assert raised.value.code == "unexpected_arguments"


def test_preview_evidence_is_compacted_without_reusable_historical_fingerprint() -> (
    None
):
    call = ToolCall(
        id="preview-history",
        name=POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
        arguments=_intent().to_payload(),
    )
    block = ToolResultBlock(
        call_id=call.id,
        output={
            "kind": POSTGRESQL_UPDATE_PREVIEW_EVIDENCE_KIND,
            "data": {
                "source_id": SOURCE_ID,
                "source_revision": SOURCE_REVISION,
                "resource_id": RESOURCE_ID,
                "resource_revision": RESOURCE_REVISION,
                "resource_name": "public.accounts",
                "would_affect": 1,
                "before": [{"column": "status", "value": "active"}],
                "after": [{"column": "status", "value": "inactive"}],
                "preview_fingerprint": "sha256:" + "a" * 64,
                "warnings": [],
                "trust_classification": "untrusted_data",
            },
        },
    )

    projected, redacted, full = context_module._project_historical_result(
        call,
        block,
        continuity=True,
    )

    assert projected is not None
    data = projected["data"]
    assert isinstance(data, Mapping)
    assert data["would_affect"] == 1
    assert "before" not in data
    assert "after" not in data
    assert "preview_fingerprint" not in data
    assert redacted is True
    assert full is False


@pytest.mark.integration
async def test_public_agent_preview_vertical_slice_creates_no_write_receipt(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_id = "preview-call"
    public_agent_id = "agent-preview-public"
    registration = _registration(agent_id=public_agent_id, write_access=False)
    public_resource_id = catalog_resource_id(
        registration.id,
        ResourceKind.TABLE,
        "public.accounts",
    )
    public_intent = _intent(
        source_id=registration.id,
        resource_id=public_resource_id,
    )
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id=call_id,
                        name=POSTGRESQL_UPDATE_PREVIEW_TOOL_NAME,
                        arguments=public_intent.to_payload(),
                    ),
                ),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="The update was previewed; no database change was made.",
            ),
        )
    )
    agent = await Agent.create(
        "postgresql-preview-public",
        root=tmp_path,
        model=provider,
        model_profile=provider.model_profile,
        clock=lambda: NOW,
        id_factory=lambda prefix: (
            public_agent_id if prefix == "agent" else f"{prefix}-preview-public"
        ),
    )
    assert agent.id == public_agent_id
    snapshot = postgresql_module._catalog_snapshot(
        registration,
        DiscoveryRequest(
            agent_id=agent.id,
            source_id=registration.id,
            sync_id="catalog-sync-preview-public",
            requested_at=NOW,
        ),
        _structure(),
        NOW,
    )
    await agent._embedded._store.register_source(registration)
    await agent._embedded._store.commit_snapshot(snapshot)
    enabled = await agent.set_source_write_access(registration.id, True)
    assert enabled.configuration["write_access"] is True
    connection = _Connection()
    _patch_io(monkeypatch, connection)
    try:
        result = await agent.run("Preview changing account 42 to inactive.")
        assert result.final_text == (
            "The update was previewed; no database change was made."
        )
        system_text = "\n".join(
            block.text
            for message in provider.requests[0].messages
            if message.role is MessageRole.SYSTEM
            for block in message.content
            if isinstance(block, TextBlock)
        )
        assert "PostgreSQL update preview is read-only evidence only" in system_text
        assert "database mutation remains unavailable" in system_text
        tool_results = tuple(
            block
            for request in provider.requests
            for message in request.messages
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock)
        )
        assert len(tool_results) == 1
        assert tool_results[0].is_error is False
        data = tool_results[0].output["data"]
        assert isinstance(data, Mapping)
        assert data["would_affect"] == 1
        assert (
            await agent._embedded._store.load_database_write_receipt_for_call(
                agent.id,
                result.run_id,
                call_id,
            )
            is None
        )
        assert not any(
            entry[0] == "fetch" and str(entry[1]).lstrip().startswith("UPDATE")
            for entry in connection.log
        )
        provider.assert_consumed()
    finally:
        await agent.close()
