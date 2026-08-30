"""Opt-in live-model evidence for one frozen Stage D2 inbox outcome.

The source, destination, schedule, outcome contract, and capability ceiling are
all established deterministically. The one authorized live scheduled run must
choose the admitted SQLite source and exact CSV artifact capability. This file
is never part of the deterministic gate.
"""

from __future__ import annotations

import asyncio
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal, InvalidOperation
from pathlib import Path

import pytest
from _workspace_support import workspace_for

from daita import (
    Agent,
    ArtifactRequirement,
    InboxView,
    IntervalSchedule,
    LoopLimits,
    MisfirePolicy,
    OutcomeContract,
    ReportingMode,
    ScheduledRoutineDraft,
    SQLiteSource,
    create_llm_provider,
)
from daita.artifacts.models import ArtifactAuthorship
from daita.domains.data.export_capabilities import (
    SQLITE_TABULAR_EXPORT_CAPABILITY_ID,
    SQLITE_TABULAR_EXPORT_TOOL_NAME,
)
from daita.llm.models import (
    FinishReason,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
)
from daita.llm.pricing import CostEstimate
from daita.llm.profiles import reviewed_model_profile
from daita.llm.providers.mock import MockModelProvider
from daita.loop.models import LoopExitKind, RunOrigin

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


def _artifact_contract() -> OutcomeContract:
    requirement = ArtifactRequirement(
        required=True,
        minimum_count=1,
        maximum_count=1,
        allowed_media_types=("text/csv",),
        allowed_authorships=(ArtifactAuthorship.EXACT_SOURCE_DATA,),
        allowed_producer_capability_ids=(SQLITE_TABULAR_EXPORT_CAPABILITY_ID,),
        maximum_artifact_bytes=8 * 1024 * 1024,
        maximum_total_bytes=8 * 1024 * 1024,
        maximum_sensitivity=ModelSensitivity.INTERNAL,
    )
    return OutcomeContract(
        require_terminal_conclusion=True,
        artifact_requirements=(requirement,),
        maximum_total_artifact_bytes=8 * 1024 * 1024,
        maximum_effective_sensitivity=ModelSensitivity.INTERNAL,
        require_current_run_provenance=True,
        require_exact_source_bindings=True,
    )


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
    provider_name = model_id.partition(":")[0]
    if provider_name not in _API_PROVIDERS:
        pytest.fail(
            f"{_MODEL_ID} must name an API-backed provider: "
            + ", ".join(sorted(_API_PROVIDERS))
        )
    profile = reviewed_model_profile(model_id)
    if profile is None or not profile.supports_tools:
        pytest.fail(f"{_MODEL_ID} must name one reviewed tool-capable model")
    api_key = os.environ.get(_MODEL_KEY)
    if api_key is None or not api_key.strip():
        pytest.fail(f"{_MODEL_KEY} must be set for the authorized live test")
    live = create_llm_provider(
        model_id,
        api_key=api_key,
        max_output_tokens=min(profile.max_output_tokens, 1_024),
    )
    cost_limit = _cost_limit()
    limits = LoopLimits(
        max_steps=8,
        max_total_tokens=12_000,
        max_wall_time_seconds=120,
        max_estimated_cost_usd=cost_limit,
    )
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
        (destination,) = await agent.distribution_destinations(
            origin.conversation_id,
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
        )
        now = datetime.now(UTC)
        proposal = await agent.propose_routine(
            ScheduledRoutineDraft(
                origin_run_id=origin.run_id,
                title="Stage D2 live CSV outcome",
                authorized_instruction=(
                    "Create one exact CSV artifact from the admitted SQLite table "
                    "stage_d2_probe. Read the scheduled row and include probe_name "
                    "and token. Use the admitted exact SQLite export capability; do "
                    "not answer from schema or copy source rows through model text."
                ),
                schedule=IntervalSchedule(3_600, now + timedelta(hours=1)),
                misfire_policy=MisfirePolicy.LATEST_ONLY,
                reporting_mode=ReportingMode.ALWAYS,
                precheck=None,
                allowed_source_ids=(source.id,),
                allowed_connector_binding_ids=(),
                allowed_resource_ids=(resource.id,),
                allowed_capability_ids=(SQLITE_TABULAR_EXPORT_CAPABILITY_ID,),
                sensitivity_ceiling=ModelSensitivity.INTERNAL,
                outcome_contract=_artifact_contract(),
                distribution_destination_id=destination.destination_id,
                eligible_model_routes=(live.provider_id,),
                per_run_max_tokens=limits.max_total_tokens,
                per_run_max_cost_usd=cost_limit,
                cumulative_max_tokens=limits.max_total_tokens * 2,
                cumulative_max_cost_usd=cost_limit * 2,
                cumulative_max_attempts=2,
                cumulative_max_occurrences=2,
                maximum_consecutive_failures=1,
                expires_at=now + timedelta(days=1),
            )
        )
        created = await agent.create_routine(proposal)
        await agent.run_routine_now(
            created.routine_id,
            expected_revision=created.revision,
        )
        deadline = asyncio.get_running_loop().time() + 180
        inbox: tuple[InboxView, ...] = ()
        while asyncio.get_running_loop().time() < deadline:
            inbox = await agent.inbox(conversation_id=origin.conversation_id)
            if inbox:
                break
            await asyncio.sleep(0.05)
        assert len(inbox) == 1
        delivery = inbox[0]
        assert delivery.failure_code is None
        (reference,) = delivery.artifact_references
        assert reference.producer_capability_id == (SQLITE_TABULAR_EXPORT_CAPABILITY_ID)
        assert reference.authorship is ArtifactAuthorship.EXACT_SOURCE_DATA
        assert reference.media_type == "text/csv"
        assert delivery.resulting_run_id is not None
        transcript = await agent.transcript(delivery.resulting_run_id)
        assert transcript.run.origin is RunOrigin.SCHEDULED_ROUTINE
        scope = transcript.run.execution_scope
        assert scope is not None
        assert scope.allowed_source_ids == (source.id,)
        assert scope.allowed_resource_ids == (resource.id,)
        assert scope.allowed_capability_ids == (SQLITE_TABULAR_EXPORT_CAPABILITY_ID,)
        calls = tuple(
            call for message in transcript.messages for call in message.tool_calls
        )
        export_calls = tuple(
            call for call in calls if call.name == SQLITE_TABULAR_EXPORT_TOOL_NAME
        )
        assert len(export_calls) == 1
        assert export_calls[0].arguments["source_id"] == source.id
        assert export_calls[0].arguments["format"] == "csv"
        payload = await agent.read_artifact(reference.artifact_id)
        assert _TOKEN.encode("utf-8") in payload.content
    finally:
        await agent.close()
