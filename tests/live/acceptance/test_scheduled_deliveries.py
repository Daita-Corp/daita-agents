"""Opt-in live-model evidence for one Stage D2 inbox outcome.

The source, schedule and routine are established deterministically. The one
authorized live scheduled run must choose the named SQLite source and the exact
CSV export. This file is never part of the deterministic gate.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest

from daita import (
    Agent,
    DeliveryView,
    IntervalSchedule,
    LoopLimits,
    RoutineIntent,
    SQLiteSource,
)
from daita.artifacts.models import ArtifactAuthorship
from daita.domains.data.export_capabilities import (
    DATA_EXPORT_TABULAR_CAPABILITY_ID,
)
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelUsage,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind, RunOrigin
from tests.support.live_billing import cost_cap, live_provider, subscription_billing
from tests.support.workspace import workspace_for

_AUTHORIZATION = "DAITA_RUN_LIVE_STAGE_D2_OUTCOME"
_MODEL_ID = "DAITA_STAGE_D2_LIVE_MODEL_ID"
_MODEL_KEY = "DAITA_STAGE_D2_LIVE_LLM_API_KEY"
_MAX_COST = "DAITA_STAGE_D2_LIVE_MAX_COST_USD"
_DEFAULT_MODEL_ID = "openai:gpt-5.6-terra"
_TOKEN = "STAGE_D2_LIVE_SENTINEL_71C4A9"
_API_PROVIDERS = frozenset({"anthropic", "gemini", "grok", "openai"})

pytestmark = [
    pytest.mark.acceptance,
    pytest.mark.integration,
    pytest.mark.requires_llm,
    pytest.mark.skipif(
        os.environ.get(_AUTHORIZATION) != "1",
        reason=(
            f"set {_AUTHORIZATION}=1 only after explicitly authorizing one live "
            f"Stage D2 scheduled model run capped by {_MAX_COST}"
        ),
    ),
]


def _cost_limit() -> Decimal:
    raw = os.environ.get(_MAX_COST, "0.10")
    try:
        value = Decimal(raw)
    except InvalidOperation:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    if not value.is_finite() or value <= 0:
        pytest.fail(f"{_MAX_COST} must be a finite positive decimal")
    return value


async def test_live_model_selects_frozen_sqlite_csv_for_inbox_outcome(
    tmp_path: Path,
) -> None:
    database = tmp_path / "stage-d2-live.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE stage_d2_probe (probe_name TEXT, token TEXT)")
        connection.execute(
            "INSERT INTO stage_d2_probe VALUES (?, ?)",
            ("scheduled", _TOKEN),
        )

    root = tmp_path / "state"
    seed_provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Stage D2 live origin established.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
        ),
        provider_id="mock:stage-d2-live-origin",
        complete_pricing=True,
    )
    seed = await Agent.create(
        "stage-d2-live",
        root=root,
        workspace=workspace_for(root),
        model=seed_provider,
        model_profile=seed_provider.model_profile,
    )
    try:
        source = await seed.attach(SQLiteSource(database, name="Stage D2 live probe"))
        (resource,) = await seed.list_catalog_resources(source_id=source.id)
        origin = await seed.run("Establish the foreground authorization context.")
        assert origin.kind is LoopExitKind.COMPLETED
        assert origin.conversation_id is not None
    finally:
        await seed.close()

    model_id = os.environ.get(_MODEL_ID, _DEFAULT_MODEL_ID)
    if not subscription_billing():
        if model_id.partition(":")[0] not in _API_PROVIDERS:
            pytest.fail(
                f"{_MODEL_ID} must name an API-backed provider: "
                + ", ".join(sorted(_API_PROVIDERS))
            )
        api_key = os.environ.get(_MODEL_KEY)
        if api_key is None or not api_key.strip():
            pytest.fail(f"{_MODEL_KEY} must be set for the authorized live test")
    profile, live = live_provider(
        model_id,
        api_key=lambda: os.environ[_MODEL_KEY],
        max_output_tokens=1_024,
    )
    cost_limit = cost_cap(_cost_limit())
    limits = LoopLimits(max_estimated_cost_usd=cost_limit)
    agent = await Agent.open(
        "stage-d2-live",
        root=root,
        workspace=workspace_for(root),
        model=live,
        model_profile=profile,
        limits=limits,
    )
    try:
        assert origin.conversation_id is not None
        created = await agent.create_routine(
            RoutineIntent(
                title="Stage D2 live CSV outcome",
                prompt=(
                    "Create one exact CSV artifact from the admitted SQLite table "
                    "stage_d2_probe. Read the scheduled row and include probe_name "
                    "and token. Use the admitted exact relational export capability; do "
                    "not answer from schema or copy source rows through model text."
                ),
                schedule=IntervalSchedule(
                    3_600, datetime.now(UTC) + timedelta(hours=1)
                ),
                sources=(source.id,),
            ),
            conversation_id=origin.conversation_id,
        )
        await agent.run_routine_now(
            created.routine_id,
            expected_revision=created.revision,
        )
        deadline = asyncio.get_running_loop().time() + 180
        deliveries: tuple[DeliveryView, ...] = ()
        while asyncio.get_running_loop().time() < deadline:
            deliveries = await agent.list_deliveries(
                conversation_id=origin.conversation_id
            )
            if deliveries:
                break
            await asyncio.sleep(0.05)
        assert len(deliveries) == 1
        delivery = deliveries[0]
        assert delivery.failure_code is None
        (reference,) = delivery.artifact_references
        assert reference.producer_capability_id == (DATA_EXPORT_TABULAR_CAPABILITY_ID)
        assert reference.authorship is ArtifactAuthorship.EXACT_SOURCE_DATA
        assert reference.media_type == "text/csv"
        assert delivery.resulting_run_id is not None
        transcript = await agent.transcript(delivery.resulting_run_id)
        assert transcript.run.origin is RunOrigin.SCHEDULED_ROUTINE
        scope = transcript.run.execution_scope
        assert scope is not None
        # Fixed at run start: the named source and every non-excluded capability.
        assert scope.allowed_source_ids == (source.id,)
        assert resource.id in scope.allowed_resource_ids
        assert DATA_EXPORT_TABULAR_CAPABILITY_ID in scope.allowed_capability_ids
        calls = tuple(
            call for call, _ in await agent.tool_evidence(delivery.resulting_run_id)
        )
        export_calls = tuple(
            call
            for call in calls
            if call.name == "data_query" and "output" in call.arguments
        )
        assert len(export_calls) == 1
        assert export_calls[0].arguments["source_id"] == source.id
        assert export_calls[0].arguments["output"] == "csv"
        payload = await agent.read_artifact(reference.artifact_id)
        assert _TOKEN.encode("utf-8") in payload.content
    finally:
        try:
            await agent.close()
        finally:
            await live.close()
