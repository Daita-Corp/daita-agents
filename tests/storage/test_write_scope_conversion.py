"""Upgrades convert stored write scopes; old write receipts keep decoding and resolving."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from daita import Agent
from daita._json import FrozenJsonObject
from daita.capabilities import ApprovalDecision, ApprovalRequest, EffectOutcome
from daita.storage.home_migrations import revision_0004
from daita.storage.sqlite_codecs.receipts import encode_receipt as encode_receipt
from daita.storage.sqlite_records import (
    EffectResolutionDecision,
    relational_write_authorization_fingerprint,
)
from tests.support.native_writes import (
    TABLE,
    ScriptedResearchModel,
    create_fixture,
    scheduled_write_script,
)
from tests.support.workspace import workspace_for


def _old_scope(operations: list[str], revision: str) -> str:
    return json.dumps(
        {
            "__record__": "RelationalWriteScope",
            "fields": {
                "resource_revision": revision,
                "allowed_operations": operations,
                "allowed_insert_columns": (
                    ["domain", "name"] if "upsert" in operations else []
                ),
                "allowed_update_columns": ["name"],
                "key_columns": ["domain"],
                "generated_identity_columns": (
                    ["id"] if "upsert" in operations else []
                ),
                "max_rows": 25,
            },
        }
    )


def _state_db(root: Path) -> Path:
    return next(root.rglob("state.db"))


def _current_revision(connection: sqlite3.Connection) -> str:
    """The revision the catalog holds for the one table these homes write."""

    for (encoded,) in connection.execute("SELECT data FROM snapshots"):
        for resource in json.loads(encoded)["fields"]["resources"]:
            if resource["fields"]["native_identity"] == TABLE:
                return resource["fields"]["current_revision"]
    raise AssertionError(f"{TABLE} is not in the catalog")


async def _home_with_stored_scope(
    tmp_path, monkeypatch, operations: list[str], revision: str | None = None
):
    agent, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    source_id = batch["source_id"]
    await agent.close()
    with sqlite3.connect(_state_db(tmp_path)) as connection:
        connection.execute(
            "UPDATE relational_write_scopes SET data = ?, "
            "authorization_fingerprint = ?",
            (
                _old_scope(operations, revision or _current_revision(connection)),
                "sha256:" + "7" * 64,
            ),
        )
    return source_id


def _convert(root: Path) -> None:
    with sqlite3.connect(_state_db(root)) as connection:
        revision_0004._convert_write_scopes(connection)
        connection.commit()


async def _reopen(tmp_path, monkeypatch):
    async def approved(request: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision.APPROVE

    model = ScriptedResearchModel()
    return await Agent.open(
        "native-acceptance",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=model,
        model_profile=model.model_profile,
        approval_handler=approved,
    )


@pytest.mark.parametrize(
    ("held", "operations", "inserts"),
    (
        (["update"], ("update",), ()),
        (["upsert"], ("insert", "update"), ("domain", "name")),
        (["update", "upsert"], ("insert", "update"), ("domain", "name")),
    ),
)
async def test_stored_scopes_convert_and_never_gain_delete(
    tmp_path, monkeypatch, held, operations, inserts
):
    source_id = await _home_with_stored_scope(tmp_path, monkeypatch, held)
    _convert(tmp_path)
    agent = await _reopen(tmp_path, monkeypatch)
    try:
        store = agent._embedded._store
        (scope,) = await store.list_relational_write_scopes(agent.id, source_id)
        assert scope.table == TABLE
        assert scope.allowed_operations == operations
        assert "delete" not in scope.allowed_operations
        assert scope.allowed_insert_columns == inserts
        assert scope.allowed_update_columns == ("name",)
        assert scope.max_rows == 25
        registration = await store.load_source(agent.id, source_id)
        assert (
            scope.authorization_fingerprint
            == relational_write_authorization_fingerprint(
                source=registration, scope=scope
            )
        )
    finally:
        await agent.close()


async def test_a_scope_whose_table_left_the_catalog_is_dropped(tmp_path, monkeypatch):
    source_id = await _home_with_stored_scope(tmp_path, monkeypatch, ["update"])
    with sqlite3.connect(_state_db(tmp_path)) as connection:
        connection.execute(
            "UPDATE relational_write_scopes SET resource_id = ?",
            ("catalog-resource:sha256:" + "9" * 64,),
        )
    _convert(tmp_path)
    agent = await _reopen(tmp_path, monkeypatch)
    try:
        assert (
            await agent._embedded._store.list_relational_write_scopes(
                agent.id, source_id
            )
            == ()
        )
    finally:
        await agent.close()


async def test_a_scope_already_refused_as_stale_stays_refused(tmp_path, monkeypatch):
    """Conversion must not restore authority the old code had already withdrawn."""

    source_id = await _home_with_stored_scope(
        tmp_path, monkeypatch, ["update"], revision="sha256:" + "5" * 64
    )
    _convert(tmp_path)
    agent = await _reopen(tmp_path, monkeypatch)
    try:
        inspection = await agent.inspect_source_permissions(source_id)
        assert inspection.state.relational_write_scopes == ()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("kind", "payload"),
    (
        ("data.update_rows", {"affected_rows": None, "resource_id": "r"}),
        ("data.upsert_rows", {"inserted_count": None, "input_count": 1}),
    ),
)
async def test_an_uncertain_receipt_from_a_removed_tool_stays_readable_and_resolves(
    tmp_path, monkeypatch, kind, payload
):
    agent, provider, db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    db.commit_error = ConnectionError("lost")
    provider.replace_script(scheduled_write_script(batch))
    await agent.run("Write it.")
    (uncertain,) = await agent.list_effects(unresolved_only=True)
    assert uncertain.outcome is EffectOutcome.UNCERTAIN
    await agent.close()

    # Rewrite the receipt as the removed tool stored it.
    with sqlite3.connect(_state_db(tmp_path)) as connection:
        connection.execute(
            "UPDATE effect_receipts SET data = ?",
            (
                encode_receipt(
                    replace(
                        uncertain,
                        receipt_kind=kind,
                        capability_id=kind,
                        payload=FrozenJsonObject.from_mapping(payload),
                    )
                ),
            ),
        )

    agent = await _reopen(tmp_path, monkeypatch)
    try:
        (listed,) = await agent.list_effects(unresolved_only=True)
        assert (listed.receipt_kind, listed.capability_id) == (kind, kind)
        resolved = await agent.resolve_effect(
            listed.receipt_id,
            expected_digest=listed.receipt_digest,
            decision=EffectResolutionDecision.CLOSE_WITHOUT_RETRY,
            note="Checked the table by hand; nothing to repeat.",
        )
        assert resolved.resolution is not None
        assert resolved.outcome is EffectOutcome.UNCERTAIN
        assert await agent.list_effects(unresolved_only=True) == ()
    finally:
        await agent.close()
