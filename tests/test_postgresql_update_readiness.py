from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from daita import Agent, PostgreSQLSource, PostgreSQLUpdateReadiness
from daita._json import canonical_json
from daita.adapters import postgresql as postgresql_module
from daita.adapters import postgresql_write as write_module
from daita.adapters.models import SourceRegistration
from daita.catalog.models import ResourceKind, TabularColumn
from daita.domains.data.sql import (
    ResourceSchema,
    validate_postgresql_update_scope,
)
from daita.security import SecretReference

NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)
SOURCE_ID = "source:sha256:" + "a" * 64
RESOURCE_ID = "catalog-resource:sha256:" + "b" * 64
RESOURCE_REVISION = "sha256:" + "c" * 64


def _table() -> postgresql_module._TableStructure:
    return postgresql_module._TableStructure(
        schema="canary",
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
        ),
        indexes=(),
    )


TABLE = _table()
SOURCE_REVISION = (
    "catalog:sha256:"
    + sha256(
        canonical_json({"relationships": (), "tables": (TABLE.payload(),)}).encode()
    ).hexdigest()
)


def _resource(**changes: object) -> ResourceSchema:
    values: dict[str, object] = {
        "resource_id": RESOURCE_ID,
        "source_id": SOURCE_ID,
        "name": "accounts",
        "aliases": ("canary.accounts",),
        "columns": ("account_id", "status"),
        "revision": RESOURCE_REVISION,
        "source_revision": SOURCE_REVISION,
        "resource_kind": "table",
        "writable": True,
        "primary_key_columns": ("account_id",),
        "column_nullability": (("account_id", False), ("status", False)),
        "column_type_provenance": (
            ("account_id", "pg_catalog", "int8"),
            ("status", "pg_catalog", "text"),
        ),
        "updatable_columns": ("status",),
    }
    values.update(changes)
    return ResourceSchema(**values)  # type: ignore[arg-type]


def _registration(*, write_access: bool) -> SourceRegistration:
    registration = SourceRegistration.build(
        agent_id="agent-readiness",
        adapter_id="postgresql",
        native_identity="postgresql:readiness",
        display_name="Readiness PostgreSQL",
        configuration={
            "database": "fixture",
            "host": "secret-host.invalid",
            "port": 5432,
            "schemas": ("canary",),
            "ssl_mode": "require",
            "username": "secret-role",
            "write_access": write_access,
        },
        attached_at=NOW,
    )
    object.__setattr__(registration, "id", SOURCE_ID)
    return registration


class _Sources:
    def __init__(self, registration: SourceRegistration) -> None:
        self.registration = registration

    async def register_source(self, registration: SourceRegistration):
        self.registration = registration
        return registration

    async def load_source(self, agent_id: str, source_id: str):
        if agent_id == self.registration.agent_id and source_id == SOURCE_ID:
            return self.registration
        return None

    async def list_sources(self, agent_id: str):
        return (self.registration,) if agent_id == self.registration.agent_id else ()

    async def detach_source(self, *args: object):
        raise AssertionError(args)


class _Catalog:
    def __init__(
        self,
        resource: ResourceSchema | None = None,
        *,
        scope_issue: tuple[str, str] | None = None,
    ) -> None:
        self.resource = resource or _resource()
        self.scope_issue = scope_issue

    async def resource_schemas(self, agent_id: str, source_id: str):
        assert agent_id == "agent-readiness"
        return (self.resource,) if source_id == SOURCE_ID else ()

    async def postgresql_update_scope_issue(self, *args: object):
        del args
        return self.scope_issue


class _Transaction:
    def __init__(self, log: list[str]) -> None:
        self.log = log

    async def start(self) -> None:
        self.log.append("start")

    async def commit(self) -> None:
        self.log.append("commit")

    async def rollback(self) -> None:
        self.log.append("rollback")


class _Connection:
    def __init__(self, facts: dict[str, object]) -> None:
        self.facts = facts
        self.log: list[str] = []

    def transaction(self, **kwargs: object) -> _Transaction:
        assert kwargs == {"isolation": "repeatable_read", "readonly": True}
        return _Transaction(self.log)

    async def execute(self, sql: str, *parameters: object) -> None:
        assert "GRANT" not in sql and "CREATE ROLE" not in sql
        self.log.append("configure")

    async def fetchrow(self, sql: str, *parameters: object, **kwargs: object):
        assert "update_preview_guardrails" in sql
        assert parameters == ("canary", "accounts", ["status"])
        self.log.append("guardrails")
        return self.facts

    async def close(self) -> None:
        self.log.append("close")

    def terminate(self) -> None:
        self.log.append("terminate")


def _facts(**changes: object) -> dict[str, object]:
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


def test_guardrail_query_uses_postgresql_current_user_keyword() -> None:
    assert "pg_catalog.current_user" not in write_module._WRITE_GUARDRAILS_SQL
    assert "role.rolname = current_user" in write_module._WRITE_GUARDRAILS_SQL


async def _readiness(
    monkeypatch: pytest.MonkeyPatch,
    *,
    write_access: bool,
    facts: dict[str, object],
    resource: ResourceSchema | None = None,
    source_revision: str = SOURCE_REVISION,
) -> tuple[PostgreSQLUpdateReadiness, _Connection]:
    connection = _Connection(facts)

    async def connect(*args: object, **kwargs: object):
        del args, kwargs
        return connection

    async def load_structure(*args: object, **kwargs: object):
        del args, kwargs
        return postgresql_module.PostgreSQLStructure(
            tables=(TABLE,),
            relationships=(),
            source_revision=source_revision,
        )

    monkeypatch.setattr(write_module, "_connect", connect)
    monkeypatch.setattr(write_module, "_load_structure", load_structure)
    backend = write_module.PostgreSQLUpdatePreviewBackend(
        _Sources(_registration(write_access=write_access)),
        _Catalog(
            resource,
            scope_issue=(
                None
                if write_access
                else (
                    "resource_update_not_allowed",
                    "The requested resource is not authorized for PostgreSQL updates.",
                )
            ),
        ),
    )
    result = await backend.postgresql_update_readiness(
        agent_id="agent-readiness",
        source_id=SOURCE_ID,
        resource_id=RESOURCE_ID,
        assignment_columns=("status",),
    )
    return result, connection


async def test_missing_exact_scope_fails_before_native_readiness_io(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, connection = await _readiness(
        monkeypatch,
        write_access=False,
        facts=_facts(),
    )

    assert result.ready_for_preview is False
    assert result.write_access is False
    assert result.proves_execution is False
    assert result.rejection_codes == ("resource_update_not_allowed",)
    assert result.remediation_categories == ("configure_source_permissions",)
    assert result.privileges["requested_columns_update"] is None
    assert tuple(field.name for field in fields(result)) == (
        "source_id",
        "resource_id",
        "assignment_columns",
        "write_access",
        "ready_for_preview",
        "proves_execution",
        "role_attributes",
        "privileges",
        "relation",
        "rejection_codes",
        "remediation_categories",
    )
    rendered = canonical_json(result.to_mapping())
    assert "secret-host" not in rendered
    assert "secret-role" not in rendered
    assert "SELECT" not in rendered
    assert "status value" not in rendered
    assert connection.log == []


async def test_readiness_rejects_powerful_roles_and_missing_column_grant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _connection = await _readiness(
        monkeypatch,
        write_access=True,
        facts=_facts(role_superuser=True, can_update_columns=False),
    )

    assert result.ready_for_preview is False
    assert result.rejection_codes == (
        "write_role_superuser",
        "write_privilege_column_update_missing",
    )
    assert result.remediation_categories == (
        "use_least_privileged_role",
        "grant_column_update_externally",
    )


async def test_readiness_passes_only_for_enabled_exact_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _connection = await _readiness(
        monkeypatch,
        write_access=True,
        facts=_facts(),
    )

    assert result.ready_for_preview is True
    assert result.rejection_codes == ()
    assert result.assignment_columns == ("status",)


@pytest.mark.parametrize(
    ("resource", "columns", "expected_code"),
    (
        (_resource(primary_key_columns=()), ("status",), "write_primary_key_required"),
        (
            _resource(primary_key_columns=("account_id", "status")),
            ("status",),
            "write_primary_key_required",
        ),
        (_resource(), ("account_id",), "write_assignment_invalid"),
        (_resource(), ("missing",), "write_assignment_invalid"),
        (
            _resource(updatable_columns=()),
            ("status",),
            "write_assignment_invalid",
        ),
        (
            _resource(
                column_type_provenance=(
                    ("account_id", "pg_catalog", "int8"),
                    ("status", "custom", "status_type"),
                )
            ),
            ("status",),
            "write_assignment_invalid",
        ),
    ),
)
def test_readiness_scope_validation_fails_closed_on_catalog_shape(
    resource: ResourceSchema,
    columns: tuple[str, ...],
    expected_code: str,
) -> None:
    result = validate_postgresql_update_scope(
        SOURCE_ID,
        RESOURCE_ID,
        columns,
        resources=(resource,),
    )

    assert result.valid is False
    assert result.issue_codes == (expected_code,)


@pytest.mark.parametrize(
    ("changes", "expected_code", "expected_remediation"),
    (
        (
            {"row_level_security": True},
            "write_relation_rls_enabled",
            "select_relation_without_rls",
        ),
        (
            {"has_user_triggers": True},
            "write_relation_user_triggers",
            "select_relation_without_user_triggers",
        ),
        (
            {"has_rewrite_rules": True},
            "write_relation_rewrite_rules",
            "select_relation_without_rewrite_rules",
        ),
        (
            {"is_partition": True},
            "write_relation_partitioned",
            "select_supported_base_table",
        ),
        (
            {"role_bypass_rls": True},
            "write_role_bypass_rls",
            "use_least_privileged_role",
        ),
        (
            {"can_select_table": False},
            "write_privilege_table_select_missing",
            "grant_table_select_externally",
        ),
    ),
)
async def test_readiness_normalizes_relation_role_and_privilege_rejections(
    monkeypatch: pytest.MonkeyPatch,
    changes: dict[str, object],
    expected_code: str,
    expected_remediation: str,
) -> None:
    result, _connection = await _readiness(
        monkeypatch,
        write_access=True,
        facts=_facts(**changes),
    )

    assert expected_code in result.rejection_codes
    assert expected_remediation in result.remediation_categories
    assert result.ready_for_preview is False


async def test_readiness_stale_live_structure_is_bounded_and_requests_refresh(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, connection = await _readiness(
        monkeypatch,
        write_access=True,
        facts=_facts(),
        source_revision="catalog:sha256:" + "d" * 64,
    )

    assert result.rejection_codes == ("write_resource_not_writable",)
    assert result.remediation_categories == ("refresh_catalog",)
    assert result.role_attributes["superuser"] is None
    assert "guardrails" not in connection.log
    assert connection.log[-2:] == ["commit", "close"]


async def test_readiness_connection_failure_omits_raw_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def failed_connect(*args: object, **kwargs: object):
        del args, kwargs
        raise RuntimeError(
            "password=raw-secret host=private-db.example SQL=SELECT sensitive"
        )

    monkeypatch.setattr(write_module, "_connect", failed_connect)
    backend = write_module.PostgreSQLUpdatePreviewBackend(
        _Sources(_registration(write_access=True)),
        _Catalog(),
    )

    result = await backend.postgresql_update_readiness(
        agent_id="agent-readiness",
        source_id=SOURCE_ID,
        resource_id=RESOURCE_ID,
        assignment_columns=("status",),
    )

    assert result.rejection_codes == ("write_readiness_unavailable",)
    assert result.remediation_categories == ("check_connection_and_credentials",)
    rendered = canonical_json(result.to_mapping())
    assert "raw-secret" not in rendered
    assert "private-db" not in rendered
    assert "SELECT sensitive" not in rendered


async def test_readiness_rejects_unavailable_source_before_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection_attempted = False

    async def forbidden_connect(*args: object, **kwargs: object):
        nonlocal connection_attempted
        del args, kwargs
        connection_attempted = True
        raise AssertionError("unavailable source must be rejected before connection")

    monkeypatch.setattr(write_module, "_connect", forbidden_connect)
    backend = write_module.PostgreSQLUpdatePreviewBackend(
        _Sources(_registration(write_access=False)),
        _Catalog(),
    )

    result = await backend.postgresql_update_readiness(
        agent_id="wrong-agent",
        source_id=SOURCE_ID,
        resource_id=RESOURCE_ID,
        assignment_columns=("status",),
    )

    assert connection_attempted is False
    assert result.ready_for_preview is False
    assert result.write_access is False
    assert result.relation["catalog_admitted"] is False
    assert result.rejection_codes == ("write_source_not_available",)
    assert result.remediation_categories == ("attach_active_postgresql_source",)


@pytest.mark.parametrize(
    "variant",
    ("wrong_adapter", "inactive"),
)
async def test_readiness_rejects_wrong_adapter_or_inactive_source_before_connection(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    registration = _registration(write_access=False)
    if variant == "wrong_adapter":
        object.__setattr__(registration, "adapter_id", "sqlite")
    else:
        object.__setattr__(registration, "detached_at", NOW)

    async def forbidden_connect(*args: object, **kwargs: object):
        del args, kwargs
        raise AssertionError("ineligible source must be rejected before connection")

    monkeypatch.setattr(write_module, "_connect", forbidden_connect)
    backend = write_module.PostgreSQLUpdatePreviewBackend(
        _Sources(registration),
        _Catalog(),
    )

    result = await backend.postgresql_update_readiness(
        agent_id="agent-readiness",
        source_id=SOURCE_ID,
        resource_id=RESOURCE_ID,
        assignment_columns=("status",),
    )

    assert result.relation["catalog_admitted"] is False
    assert result.rejection_codes == ("write_source_not_available",)


@pytest.mark.parametrize(
    "assignment_columns",
    (
        (),
        ("status", "status"),
        ("status\x00hidden",),
        tuple(f"column_{index}" for index in range(33)),
    ),
)
async def test_readiness_rejects_unbounded_assignment_scope_before_source_io(
    assignment_columns: tuple[str, ...],
) -> None:
    backend = write_module.PostgreSQLUpdatePreviewBackend(
        _Sources(_registration(write_access=False)),
        _Catalog(),
    )

    with pytest.raises(ValueError, match="one through 32 distinct bounded names"):
        await backend.postgresql_update_readiness(
            agent_id="agent-readiness",
            source_id=SOURCE_ID,
            resource_id=RESOURCE_ID,
            assignment_columns=assignment_columns,
        )


async def test_first_party_postgresql_attachment_cannot_enable_writes(
    tmp_path: Path,
) -> None:
    agent = await Agent.create("read-only-attachment", root=tmp_path)
    try:
        source = PostgreSQLSource(
            host="db.example.test",
            database="warehouse",
            username="writer",
            credential=SecretReference.environment("FIXTURE_PASSWORD"),
            write_access=True,
        )
        with pytest.raises(ValueError, match="attachment is read-only"):
            await agent.attach(source)
    finally:
        await agent.close()


async def test_public_agent_readiness_delegates_exact_scope_without_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected, _connection = await _readiness(
        monkeypatch,
        write_access=True,
        facts=_facts(),
    )
    calls: list[dict[str, object]] = []
    agent = await Agent.create("public-readiness", root=tmp_path)

    async def readiness(**kwargs: object) -> PostgreSQLUpdateReadiness:
        calls.append(kwargs)
        return expected

    monkeypatch.setattr(
        agent._embedded._postgresql_update_backend,
        "postgresql_update_readiness",
        readiness,
    )
    try:
        actual = await agent.postgresql_update_readiness(
            SOURCE_ID,
            RESOURCE_ID,
            ("status",),
        )
        with pytest.raises(TypeError, match="must be a tuple"):
            await agent.postgresql_update_readiness(
                SOURCE_ID,
                RESOURCE_ID,
                ["status"],  # type: ignore[arg-type]
            )
    finally:
        await agent.close()

    assert actual is expected
    assert calls == [
        {
            "agent_id": agent.id,
            "source_id": SOURCE_ID,
            "resource_id": RESOURCE_ID,
            "assignment_columns": ("status",),
        }
    ]
