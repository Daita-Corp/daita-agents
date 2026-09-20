"""Deterministic contract checks for the live graph-effect fixture harness."""

from __future__ import annotations

from decimal import Decimal

import pytest

from daita.loop.models import LoopLimits
from tests.support.graph_effects_live import (
    NATIVE_UPSERT_FIXTURE,
    create_native_graph_effect_fixture,
    load_native_upsert_fixture,
)
from tests.support.native_writes import ScriptedResearchModel

pytestmark = pytest.mark.unit


async def test_native_live_graph_fixture_composes_credential_free_fake_database(
    tmp_path,
    monkeypatch,
) -> None:
    payload = load_native_upsert_fixture()
    rendered = NATIVE_UPSERT_FIXTURE.read_text(encoding="utf-8").lower()
    assert payload["source"]["configuration"]["host"] == "db.fixture.test"
    assert "password" not in rendered
    assert "secret" not in rendered

    provider = ScriptedResearchModel()
    fixture = await create_native_graph_effect_fixture(
        tmp_path,
        monkeypatch,
        model=provider,
        model_profile=provider.model_profile,
        limits=LoopLimits(max_estimated_cost_usd=Decimal(1)),
    )
    try:
        assert fixture.database.rows == {}
        assert fixture.intent["source_id"] == fixture.source_id
        assert fixture.intent["resource_id"] == fixture.resource_id
        assert fixture.grant_constraints["resource_revision"] == (
            fixture.resource_revision
        )
        assert fixture.expected_row["domain"] == "graph-live.test"
        assert fixture.approvals == []
    finally:
        await fixture.agent.close()
