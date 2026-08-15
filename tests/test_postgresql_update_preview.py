from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime

import pytest

from daita.adapters import (
    postgresql as postgresql_module,
    postgresql_write as write_module,
)
from daita.adapters.models import SourceRegistration, source_registration_id
from daita.catalog.models import ResourceKind, TabularColumn
from daita.domains.data.sql import PostgreSQLUpdateIntent, ResourceSchema
from daita.security import EmptySecretProvider

NOW = datetime(2026, 8, 14, 12, 0, tzinfo=UTC)
SOURCE_ID = source_registration_id(
    "agent-preview", "postgresql", "postgresql:preview-contract"
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
    def __init__(self, *, scope_issue: tuple[str, str] | None = None) -> None:
        self.scope_issue = scope_issue

    async def resource_schemas(self, agent_id: str, source_id: str):
        del agent_id
        return (_resource(),) if source_id == SOURCE_ID else ()

    async def postgresql_update_scope_issue(
        self, agent_id, source_id, resource_id, assignment_columns
    ):
        del agent_id, source_id, resource_id, assignment_columns
        return self.scope_issue


class _Transaction:
    def __init__(self, log: list[tuple[object, ...]]) -> None:
        self.log = log

    async def start(self) -> None:
        self.log.append(("transaction.start",))

    async def commit(self) -> None:
        self.log.append(("transaction.commit",))

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
        guardrails: Mapping[str, object] | None = None,
    ) -> None:
        self.rows = rows
        self.guardrails = dict(guardrails or _guardrails())
        self.log: list[tuple[object, ...]] = []
        self.transaction_record = _Transaction(self.log)

    def transaction(self, **kwargs: object):
        self.log.append(("transaction", kwargs))
        return self.transaction_record

    async def execute(self, sql: str, *parameters: object, **kwargs: object):
        self.log.append(("execute", sql, parameters, kwargs))
        if sql.startswith("UPDATE"):
            raise AssertionError("preview must never execute UPDATE")
        return "SELECT 1"

    async def fetchrow(self, sql: str, *parameters: object, **kwargs: object):
        self.log.append(("fetchrow", sql, parameters, kwargs))
        return self.guardrails

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
        agent_id="agent-preview",
        adapter_id="postgresql",
        native_identity="postgresql:preview-contract",
        display_name="Preview PostgreSQL",
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


def _row(
    key: int,
    *,
    before: str = "active",
    within: bool = True,
    xmin: str | None = None,
):
    return {
        "__daita_primary_key_0": key,
        "__daita_before_0": before if within else None,
        "__daita_within_preview_limit": within,
        "__daita_xmin": xmin or str(700 + key),
    }


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


def _backend(*, scope_issue: tuple[str, str] | None = None):
    return write_module.PostgreSQLUpdatePreviewBackend(
        _SourceStore(_registration()),
        _Catalog(scope_issue=scope_issue),
        EmptySecretProvider(),
    )


def _patch_io(monkeypatch: pytest.MonkeyPatch, connection: _Connection) -> None:
    async def connect(*args, **kwargs):
        del args, kwargs
        return connection

    async def load_structure(*args, **kwargs):
        del args, kwargs
        return _structure()

    monkeypatch.setattr(write_module, "_connect", connect)
    monkeypatch.setattr(write_module, "_load_structure", load_structure)


async def test_bulk_preview_streams_exact_count_and_bounded_samples(monkeypatch):
    connection = _Connection(tuple(_row(index) for index in range(1, 9)))
    _patch_io(monkeypatch, connection)

    preview = await _backend().preview_update(
        agent_id="agent-preview", intent=_intent()
    )

    assert preview.matched_rows == 8
    assert len(preview.samples) == 5
    assert [sample.primary_key[0].value for sample in preview.samples] == [
        1,
        2,
        3,
        4,
        5,
    ]
    assert preview.fingerprint.target_set_sha256.startswith("sha256:")
    cursor_call = next(item for item in connection.log if item[0] == "cursor")
    assert "ORDER BY" in str(cursor_call[1])
    assert cursor_call[2] == ("active", 2)
    assert not any(
        item[0] == "execute" and str(item[1]).startswith("UPDATE")
        for item in connection.log
    )


async def test_single_row_uses_the_same_preview_contract(monkeypatch):
    connection = _Connection((_row(42),))
    _patch_io(monkeypatch, connection)
    preview = await _backend().preview_update(
        agent_id="agent-preview", intent=_intent()
    )
    assert preview.matched_rows == 1
    assert preview.samples[0].primary_key[0].value == 42


async def test_zero_row_preview_is_successful_but_warns(monkeypatch):
    connection = _Connection(())
    _patch_io(monkeypatch, connection)
    preview = await _backend().preview_update(
        agent_id="agent-preview", intent=_intent()
    )
    assert preview.matched_rows == 0
    assert preview.samples == ()
    assert preview.warnings == ("target_not_found",)


async def test_oversized_values_do_not_prevent_exact_target_count(monkeypatch):
    connection = _Connection((_row(1, within=False), _row(2)))
    _patch_io(monkeypatch, connection)
    preview = await _backend().preview_update(
        agent_id="agent-preview", intent=_intent()
    )
    assert preview.matched_rows == 2
    assert len(preview.samples) == 1
    assert "oversized_sample_values_omitted" in preview.warnings


async def test_oversized_values_use_row_version_for_drift_fingerprinting(monkeypatch):
    first_connection = _Connection((_row(1, within=False, xmin="701"),))
    _patch_io(monkeypatch, first_connection)
    first = await _backend().preview_update(agent_id="agent-preview", intent=_intent())

    second_connection = _Connection((_row(1, within=False, xmin="702"),))
    _patch_io(monkeypatch, second_connection)
    second = await _backend().preview_update(agent_id="agent-preview", intent=_intent())

    assert first.fingerprint.target_set_sha256 != second.fingerprint.target_set_sha256


async def test_guardrails_fail_closed(monkeypatch):
    connection = _Connection((_row(1),), guardrails=_guardrails(role_superuser=True))
    _patch_io(monkeypatch, connection)
    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as captured:
        await _backend().preview_update(agent_id="agent-preview", intent=_intent())
    assert captured.value.error_code == "write_guardrail_rejected"


async def test_scope_is_checked_before_source_connection(monkeypatch):
    async def forbidden_connect(*args, **kwargs):
        raise AssertionError("scope rejection must happen before source I/O")

    monkeypatch.setattr(write_module, "_connect", forbidden_connect)
    with pytest.raises(write_module.PostgreSQLUpdatePreviewError) as captured:
        await _backend(
            scope_issue=("update_column_not_allowed", "not allowed")
        ).preview_update(agent_id="agent-preview", intent=_intent())
    assert captured.value.error_code == "update_column_not_allowed"
