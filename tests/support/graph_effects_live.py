"""Local effect harnesses for opt-in live-model graph tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from daita import Agent
from daita.adapters import postgresql as pg, postgresql_write as native
from daita.adapters.models import DiscoveryRequest, SourceRegistration
from daita.capabilities import ApprovalDecision, ApprovalRequest
from daita.catalog.models import ResourceKind, Sensitivity, TabularColumn, TabularIndex
from daita.llm.models import ModelProfile
from daita.llm.protocols import ModelProvider
from daita.loop.models import LoopLimits
from tests.data.writes._upsert_support import Database
from tests.support.paths import FIXTURES_ROOT
from tests.support.workspace import workspace_for

NOW = datetime(2026, 9, 20, 12, tzinfo=UTC)
NATIVE_UPSERT_FIXTURE = FIXTURES_ROOT / "graph-effects" / "native-upsert.json"


def load_native_upsert_fixture() -> dict[str, Any]:
    """Load the reviewed, credential-free fake PostgreSQL contract."""

    payload = json.loads(NATIVE_UPSERT_FIXTURE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("native graph-effect fixture must be a JSON object")
    return cast(dict[str, Any], payload)


def _object(value: object, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    return cast(Mapping[str, Any], value)


def _sequence(value: object, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be an array")
    return cast(Sequence[Any], value)


def _text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field_name} must be non-empty text")
    return value


def _text_tuple(value: object, field_name: str) -> tuple[str, ...]:
    result = tuple(_sequence(value, field_name))
    if any(not isinstance(item, str) or not item for item in result):
        raise TypeError(f"{field_name} must contain non-empty strings")
    return cast(tuple[str, ...], result)


@dataclass(slots=True)
class NativeGraphEffectFixture:
    agent: Agent
    database: Database
    source_id: str
    resource_id: str
    resource_revision: str
    intent: dict[str, object]
    grant_constraints: dict[str, object]
    expected_row: dict[str, object]
    approvals: list[ApprovalRequest]


async def create_native_graph_effect_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    model: ModelProvider,
    model_profile: ModelProfile,
    limits: LoopLimits,
) -> NativeGraphEffectFixture:
    """Compose a real agent around fixture-backed transactional PostgreSQL I/O."""

    fixture = load_native_upsert_fixture()
    source = _object(fixture.get("source"), "source")
    table_value = _object(fixture.get("table"), "table")
    scope_value = _object(fixture.get("write_scope"), "write_scope")
    intent_value = _object(fixture.get("upsert_intent"), "upsert_intent")

    columns = tuple(
        TabularColumn(
            name=_text(column["name"], "column name"),
            native_type=_text(column["native_type"], "column native_type"),
            ordinal=ordinal,
            nullable=bool(column["nullable"]),
            native_type_namespace=_text(
                column["native_type_namespace"], "column native_type_namespace"
            ),
            native_type_name=_text(
                column["native_type_name"], "column native type name"
            ),
            primary_key_ordinal=cast(int | None, column["primary_key_ordinal"]),
            identity=bool(column["identity"]),
            updatable=bool(column["updatable"]),
            collation=cast(str | None, column["collation"]),
        )
        for ordinal, column in enumerate(
            _object(item, "table column")
            for item in _sequence(table_value.get("columns"), "table columns")
        )
    )
    indexes = tuple(
        TabularIndex(
            name=_text(index["name"], "index name"),
            kind=_text(index["method"], "index method"),
            columns=_text_tuple(index["columns"], "index columns"),
            unique=bool(index["unique"]),
            write_conflict_supported=bool(index["write_conflict_supported"]),
        )
        for index in (
            _object(item, "table index")
            for item in _sequence(table_value.get("indexes"), "table indexes")
        )
    )
    table = pg._TableStructure(
        _text(table_value.get("schema"), "table schema"),
        _text(table_value.get("name"), "table name"),
        ResourceKind.TABLE,
        columns,
        indexes,
    )
    database = Database()
    structure = pg.PostgreSQLStructure((table,), (), database.structure_revision)

    async def connect(*args: object, **kwargs: object):
        return database.connect()

    async def load_structure(*args: object, **kwargs: object):
        return replace(structure, source_revision=database.structure_revision)

    monkeypatch.setattr(native, "_connect", connect)
    monkeypatch.setattr(native, "_load_structure", load_structure)

    approvals: list[ApprovalRequest] = []

    async def approve(request: ApprovalRequest) -> ApprovalDecision:
        approvals.append(request)
        return ApprovalDecision.APPROVE

    agent = await Agent.create(
        "live-native-graph-effect",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=model,
        model_profile=model_profile,
        limits=limits,
        clock=lambda: NOW,
        approval_handler=approve,
    )
    configuration = _object(source.get("configuration"), "source configuration")
    registration = SourceRegistration.build(
        agent_id=agent.id,
        adapter_id="postgresql",
        native_identity=_text(source.get("native_identity"), "source native_identity"),
        display_name=_text(source.get("display_name"), "source display name"),
        configuration=dict(configuration),
        attached_at=NOW,
    )
    snapshot = pg._catalog_snapshot(
        registration,
        DiscoveryRequest(
            agent_id=agent.id,
            source_id=registration.id,
            sync_id="live-native-graph-effect",
            requested_at=NOW,
        ),
        structure,
        NOW,
    )
    snapshot = replace(
        snapshot,
        resources=tuple(
            replace(item, sensitivity=Sensitivity.RESTRICTED)
            for item in snapshot.resources
        ),
    )
    await agent._embedded._store.commit_snapshot(snapshot, registration=registration)
    resource = snapshot.resources[0]
    relational_scope: dict[str, object] = {
        "allowed_operations": _text_tuple(
            scope_value.get("allowed_operations"), "allowed operations"
        ),
        "allowed_insert_columns": _text_tuple(
            scope_value.get("allowed_insert_columns"), "allowed insert columns"
        ),
        "allowed_update_columns": _text_tuple(
            scope_value.get("allowed_update_columns"), "allowed update columns"
        ),
        "key_columns": _text_tuple(scope_value.get("key_columns"), "key columns"),
        "generated_identity_columns": _text_tuple(
            scope_value.get("generated_identity_columns"),
            "generated identity columns",
        ),
        "max_rows": scope_value.get("max_rows"),
    }
    permission_preview = await agent.preview_source_permissions(
        source_id=registration.id,
        read_mode="all",
        read_resource_ids=(),
        relational_write_scopes={resource.id: relational_scope},
    )
    await agent.apply_source_permissions(
        source_id=registration.id,
        confirmation_fingerprint=permission_preview.confirmation_fingerprint,
    )

    rows = tuple(
        dict(_object(item, "upsert row"))
        for item in _sequence(intent_value.get("rows"), "upsert rows")
    )
    intent: dict[str, object] = {
        "source_id": registration.id,
        "resource_id": resource.id,
        "key_columns": _text_tuple(intent_value.get("key_columns"), "key columns"),
        "insert_columns": _text_tuple(
            intent_value.get("insert_columns"), "insert columns"
        ),
        "update_columns": _text_tuple(
            intent_value.get("update_columns"), "update columns"
        ),
        "rows": rows,
    }
    grant_constraints = {
        "source_id": registration.id,
        "resource_id": resource.id,
        "resource_revision": resource.current_revision,
        **{
            key: value
            for key, value in relational_scope.items()
            if key != "allowed_operations"
        },
    }
    return NativeGraphEffectFixture(
        agent=agent,
        database=database,
        source_id=registration.id,
        resource_id=resource.id,
        resource_revision=resource.current_revision,
        intent=intent,
        grant_constraints=grant_constraints,
        expected_row=rows[0],
        approvals=approvals,
    )
