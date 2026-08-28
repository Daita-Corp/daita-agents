from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

from _workspace_support import workspace_for

from daita import (
    Agent,
    IntervalSchedule,
    MisfirePolicy,
    ReportingMode,
    RoutineState,
    SQLiteSource,
    ScheduledRoutineDraft,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
)
from daita.llm.providers.mock import MockModelProvider
from daita.llm.pricing import CostEstimate


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=20_000,
        max_output_tokens=1_000,
        supports_tools=True,
        supports_parallel_tools=True,
    )


async def test_public_routine_surface_walks_create_and_lifecycle(
    tmp_path: Path,
) -> None:
    database = tmp_path / "current.sqlite"
    connection = sqlite3.connect(database)
    connection.execute("CREATE TABLE current_value (value INTEGER NOT NULL)")
    connection.execute("INSERT INTO current_value VALUES (7)")
    connection.commit()
    connection.close()

    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Current value is 7.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
            ModelResponse(
                finish_reason=FinishReason.STOP,
                text="Current value is 7.",
                usage=ModelUsage(cost_estimate=CostEstimate.complete(Decimal("0"))),
            ),
        ),
        provider_id="mock:routines-public",
        complete_pricing=True,
    )
    agent = await Agent.create(
        "routine-public",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        source = await agent.attach(SQLiteSource(database, name="Current"))
        resource = (await agent.list_catalog_resources(source_id=source.id))[0]
        origin = await agent.run("Read the current value for a scheduled report.")
        assert origin.conversation_id is not None
        now = datetime.now(UTC)
        draft = ScheduledRoutineDraft(
            origin_run_id=origin.run_id,
            title="Current value report",
            authorized_instruction=(
                "Inspect the exact current_value resource and report its current value."
            ),
            schedule=IntervalSchedule(3_600, now + timedelta(hours=1)),
            misfire_policy=MisfirePolicy.LATEST_ONLY,
            reporting_mode=ReportingMode.ALWAYS,
            precheck=None,
            allowed_source_ids=(source.id,),
            allowed_connector_binding_ids=(),
            allowed_resource_ids=(resource.id,),
            allowed_capability_ids=("catalog.inspect",),
            sensitivity_ceiling=ModelSensitivity.INTERNAL,
            eligible_model_routes=(provider.provider_id,),
            per_run_max_tokens=1_000,
            per_run_max_cost_usd=Decimal("0"),
            cumulative_max_tokens=10_000,
            cumulative_max_cost_usd=Decimal("0"),
            cumulative_max_attempts=10,
            cumulative_max_occurrences=10,
            maximum_consecutive_failures=3,
            expires_at=now + timedelta(days=30),
        )
        proposal = await agent.propose_routine(draft)
        assert proposal.next_due_at is None
        created = await agent.create_routine(proposal)
        assert created.state is RoutineState.ACTIVE
        assert (await agent.list_routines())[0].routine_id == created.routine_id
        assert await agent.inspect_routine(created.routine_id) is not None

        paused = await agent.pause_routine(
            created.routine_id,
            expected_revision=created.revision,
        )
        resumed = await agent.resume_routine(
            paused.routine_id,
            expected_revision=paused.revision,
        )
        running = await agent.run_routine_now(
            resumed.routine_id,
            expected_revision=resumed.revision,
        )
        for _ in range(100):
            inbox = await agent.inbox(conversation_id=origin.conversation_id)
            if inbox:
                break
            await asyncio.sleep(0)
        inspection = await agent.inspect_routine(running.routine_id)
        assert inspection is not None
        assert len(inbox) == 1, (
            tuple(
                (
                    item.disposition,
                    item.failure_code,
                    item.reserved_run_id,
                    item.terminal_run_id,
                )
                for item in inspection.recent_occurrences
            ),
            len(provider.requests),
        )
        disabled = await agent.disable_routine(
            running.routine_id,
            expected_revision=inspection.routine.revision,
        )
        assert disabled.state is RoutineState.DISABLED
    finally:
        await agent.close()
