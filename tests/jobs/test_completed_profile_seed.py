"""Deterministic checks for internal profile jobs.

No model tool starts a profile job; the seed admits the internal static profile
graph directly, as migrated revision-2 profile jobs run it, and never calls a
paid API.
"""

from __future__ import annotations

import sqlite3

import pytest

from daita import Agent
from daita.capabilities import capability_contract_digest
from daita.domains.data.profile_jobs import (
    DATA_PROFILE_DOMAIN_OWNER_ID,
    DATA_PROFILE_EXECUTION_CAPABILITY_ID,
    DATA_PROFILE_FINALIZE_CAPABILITY_ID,
)
from daita.llm.providers.mock import MockModelProvider
from tests.support import job_benchmarks as _support
from tests.support.job_benchmarks import (
    TARGET_PROFILE_TABLE,
    stop_response,
)
from tests.support.workspace import workspace_for

pytestmark = [pytest.mark.integration, pytest.mark.acceptance]


async def test_completed_profile_seed_differs_from_current_source(
    tmp_path, monkeypatch
):
    provider = MockModelProvider((stop_response(),))
    monkeypatch.setattr(
        _support,
        "live_provider",
        lambda _model_id: (
            provider.model_profile,
            _support.RecordingProvider(provider),
        ),
    )
    fixture, job_id = await _support.seed_completed_profile_agent(
        tmp_path,
        "historical-profile",
        provider.provider_id,
    )
    try:
        await _support.assert_profile_result(fixture.agent, job_id)
        with sqlite3.connect(tmp_path / "historical-profile.sqlite") as connection:
            current = connection.execute(
                f"SELECT count(*), sum(nullable_code IS NULL) FROM {TARGET_PROFILE_TABLE}"
            ).fetchone()
        assert current == (7, 4)
        assert current != (_support.PROFILE_SAMPLE_ROWS, _support.PROFILE_NULL_VALUES)
    finally:
        await fixture.close()


async def test_migrated_profile_tasks_keep_their_internal_contracts(tmp_path):
    """Revision-2 conversion binds profile tasks to these exact contracts."""

    agent = await Agent.create(
        "migrated-profile-contracts",
        root=tmp_path / "home",
        workspace=workspace_for(tmp_path),
    )
    try:
        registry = agent._embedded._capabilities
        expected = {
            DATA_PROFILE_EXECUTION_CAPABILITY_ID: (
                "sha256:b6aaefda34441f44c891d703eb029e2e"
                "42f31d1e2920dead59fb2c5584d1d46c"
            ),
            DATA_PROFILE_FINALIZE_CAPABILITY_ID: (
                "sha256:bb221b7533d589dd8bc80d27dacd3ad6"
                "627d332ddfb276b62b00dff08f68178a"
            ),
        }
        for capability_id, digest in expected.items():
            capability, executor = registry.resolve_execution(capability_id)
            assert executor.executor_id == f"{capability_id}.executor"
            assert (
                capability_contract_digest(
                    capability, domain_owner_id=DATA_PROFILE_DOMAIN_OWNER_ID
                )
                == digest
            )
        assert not any(
            registry.resolve_tool(name)[1].id in expected
            for name in registry.tool_names
        )
    finally:
        await agent.close()
