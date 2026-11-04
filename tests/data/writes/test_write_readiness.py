"""Readiness reads one table's stored scope live, and never writes or grants."""

from __future__ import annotations

import pytest

from tests.support.native_writes import TABLE, create_fixture
from tests.support.write_database import write_guardrails


async def test_readiness_passes_for_a_stored_scope_on_a_writable_table(
    tmp_path, monkeypatch
):
    agent, _provider, db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    try:
        readiness = await agent.relational_write_readiness(batch["source_id"], TABLE)
        assert readiness.ready_for_preview and readiness.daita_scope_ready
        assert readiness.proves_execution is False
        assert readiness.table == TABLE
        assert readiness.allowed_operations == ("insert", "update")
        assert readiness.relation["base_table"] is True
        mapping = readiness.to_mapping()
        assert mapping["table"] == TABLE and "resource_id" not in mapping
        assert ("commit", True) in db.log and not db.inserts()
    finally:
        await agent.close()


@pytest.mark.parametrize(
    ("facts", "code"),
    (
        ({"relation_kind": "v"}, "write_relation_not_base_table"),
        ({"is_partition": True}, "write_relation_partitioned"),
        ({"has_inheritance": True}, "write_relation_inherited"),
        ({"row_level_security": True}, "write_relation_rls_enabled"),
        ({"has_rewrite_rules": True}, "write_relation_rewrite_rules"),
        ({"role_superuser": True}, "write_role_superuser"),
        ({"role_bypass_rls": True}, "write_role_bypass_rls"),
        ({"can_select_table": False}, "write_privilege_table_select_missing"),
        ({"can_update_columns": False}, "write_privilege_column_update_missing"),
        ({"can_insert_columns": False}, "write_privilege_column_insert_missing"),
    ),
)
async def test_readiness_names_what_the_table_or_role_lacks(
    tmp_path, monkeypatch, facts, code
):
    agent, _provider, db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    db.guardrails = write_guardrails(**facts)
    try:
        readiness = await agent.relational_write_readiness(batch["source_id"], TABLE)
        assert not readiness.ready_for_preview
        assert code in readiness.rejection_codes
    finally:
        await agent.close()


async def test_triggers_are_a_fact_for_the_card_not_a_rejection(tmp_path, monkeypatch):
    agent, _provider, db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    db.guardrails = write_guardrails(triggers='["audit"]')
    try:
        readiness = await agent.relational_write_readiness(batch["source_id"], TABLE)
        assert readiness.ready_for_preview
        assert readiness.relation["user_triggers"] is True
    finally:
        await agent.close()


async def test_a_table_without_a_scope_is_refused_before_connecting(
    tmp_path, monkeypatch
):
    agent, _provider, db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )
    db.log.clear()
    try:
        readiness = await agent.relational_write_readiness(
            batch["source_id"], "public.other"
        )
        assert readiness.rejection_codes == ("resource_write_not_allowed",)
        assert not db.log
    finally:
        await agent.close()


async def test_an_unknown_source_is_refused_before_connecting(tmp_path, monkeypatch):
    agent, _provider, db, *_ = await create_fixture(tmp_path, monkeypatch)
    db.log.clear()
    try:
        readiness = await agent.relational_write_readiness(
            "source:sha256:" + "f" * 64, TABLE
        )
        assert readiness.rejection_codes == ("write_source_not_available",)
        assert not db.log
    finally:
        await agent.close()


async def test_a_connection_failure_omits_raw_diagnostics(tmp_path, monkeypatch):
    from daita.adapters import postgresql_write as native

    agent, _provider, _db, *_rest, batch, _clock, _approvals = await create_fixture(
        tmp_path, monkeypatch
    )

    async def fail(*args: object, **kwargs: object) -> object:
        raise ConnectionError("postgresql://writer:hunter2@db.example.test")

    monkeypatch.setattr(native, "_connect", fail)
    try:
        readiness = await agent.relational_write_readiness(batch["source_id"], TABLE)
        assert readiness.rejection_codes == ("write_readiness_unavailable",)
        assert "hunter2" not in repr(readiness.to_mapping())
    finally:
        await agent.close()
