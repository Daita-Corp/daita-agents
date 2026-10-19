"""The work board places a real agent's jobs, routines and results in columns."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path

import pytest

from daita import (
    Agent,
    ArtifactRequirement,
    DeliveryView,
    IntervalSchedule,
    MisfirePolicy,
    OutcomeContract,
    OutcomeState,
    ReportingMode,
    ScheduledRoutineDraft,
    SQLiteSource,
    WorkBoard,
    WorkColumn,
    WorkKind,
)
from daita.artifacts.models import ArtifactAuthorship
from daita.domains.data.export_capabilities import DATA_EXPORT_TABULAR_CAPABILITY_ID
from daita.jobs.graph.models import (
    BudgetAmount,
    ControlKind,
    ControlState,
    TaskControl,
    canonical_digest,
)
from daita.llm.models import (
    FinishReason,
    ModelProfile,
    ModelResponse,
    ModelSensitivity,
    ModelUsage,
)
from daita.llm.pricing import CostEstimate
from daita.llm.providers.mock import MockModelProvider
from daita.storage.sqlite import SQLiteStateStore
from tests.support.distribution import no_artifact_outcome_contract
from tests.support.graph import GRAPH_NOW, graph_admission, with_topology
from tests.support.workspace import workspace_for

pytestmark = pytest.mark.acceptance

# Inside the seeded jobs' one-hour deadline, so the supervisor never expires them.
NOW = GRAPH_NOW + timedelta(minutes=10)
NAME = "work-board"
USAGE = ModelUsage(
    input_tokens=100,
    output_tokens=10,
    cost_estimate=CostEstimate.complete(Decimal("0")),
)


def _stop(text: str) -> ModelResponse:
    return ModelResponse(finish_reason=FinishReason.STOP, text=text, usage=USAGE)


def _profile(provider: MockModelProvider) -> ModelProfile:
    return ModelProfile(
        id=provider.provider_id,
        context_window_tokens=128_000,
        max_output_tokens=2_000,
        supports_tools=True,
        supports_parallel_tools=False,
    )


def _csv_export_contract() -> OutcomeContract:
    requirement = ArtifactRequirement(
        required=True,
        minimum_count=1,
        maximum_count=1,
        allowed_media_types=("text/csv",),
        allowed_authorships=(ArtifactAuthorship.EXACT_SOURCE_DATA,),
        allowed_producer_capability_ids=(DATA_EXPORT_TABULAR_CAPABILITY_ID,),
        maximum_artifact_bytes=1024 * 1024,
        maximum_total_bytes=1024 * 1024,
        maximum_sensitivity=ModelSensitivity.INTERNAL,
    )
    return OutcomeContract(
        require_terminal_conclusion=True,
        artifact_requirements=(requirement,),
        maximum_total_artifact_bytes=1024 * 1024,
        maximum_effective_sensitivity=ModelSensitivity.INTERNAL,
        require_current_run_provenance=True,
        require_exact_source_bindings=True,
    )


async def _seed_queued_and_blocked_jobs(state_path: Path, agent_id: str) -> None:
    """Admit a job with no ready work and a job blocked on the user's input."""

    store = await SQLiteStateStore.open(state_path)
    try:
        queued = graph_admission(job_id="job-queued", agent_id=agent_id)
        # Ready only after the test's fixed clock, so no supervisor claims it.
        later = GRAPH_NOW + timedelta(minutes=50)
        tasks = tuple(
            replace(task, not_before=later) if task.task_id == "worker" else task
            for task in queued.tasks
        )
        queued = with_topology(queued, tasks=tasks, dependencies=queued.dependencies)
        await store.admit_graph(
            replace(queued, graph=replace(queued.graph, next_ready_at=later))
        )
        await store.admit_graph(
            graph_admission(job_id="job-blocked", agent_id=agent_id)
        )
        attempt = await store.claim_graph_task(
            agent_id,
            "job-blocked",
            "worker",
            attempt_id="attempt-1",
            claim_token="claim-1",
            run_id="run-1",
            executor_id="executor-1",
            claimed_at=GRAPH_NOW,
            lease_seconds=30,
            absolute_deadline_at=GRAPH_NOW + timedelta(minutes=2),
            budget_reservations=(BudgetAmount("work_units", 2),),
        )
        assert attempt is not None
        assert await store.start_graph_attempt(
            agent_id,
            "job-blocked",
            "worker",
            "attempt-1",
            claim_token="claim-1",
            fencing_epoch=attempt.fencing_epoch,
            started_at=GRAPH_NOW + timedelta(seconds=1),
        )
        payload = {"question": "Which region should the report cover?"}
        await store.open_graph_control(
            TaskControl(
                agent_id=agent_id,
                job_id="job-blocked",
                task_id="worker",
                control_id="control-1",
                kind=ControlKind.NEEDS_INPUT,
                state=ControlState.OPEN,
                requesting_attempt_id="attempt-1",
                payload=payload,
                created_at=GRAPH_NOW + timedelta(seconds=2),
                payload_digest=canonical_digest(payload),
            ),
            claim_token="claim-1",
            fencing_epoch=attempt.fencing_epoch,
        )
    finally:
        await store.close()


async def _run_routine(
    agent: Agent,
    *,
    origin_run_id: str,
    conversation_id: str,
    title: str,
    source_id: str,
    resource_id: str,
    capability_id: str,
    outcome_contract: OutcomeContract,
    provider: MockModelProvider,
) -> DeliveryView:
    (destination,) = await agent.distribution_destinations(
        conversation_id, sensitivity_ceiling=ModelSensitivity.INTERNAL
    )
    draft = ScheduledRoutineDraft(
        origin_run_id=origin_run_id,
        title=title,
        authorized_instruction="Report the current value from the exact resource.",
        schedule=IntervalSchedule(3_600, NOW + timedelta(hours=1)),
        misfire_policy=MisfirePolicy.LATEST_ONLY,
        reporting_mode=ReportingMode.ALWAYS,
        precheck=None,
        allowed_source_ids=(source_id,),
        allowed_connector_binding_ids=(),
        allowed_resource_ids=(resource_id,),
        allowed_capability_ids=(capability_id,),
        sensitivity_ceiling=ModelSensitivity.INTERNAL,
        outcome_contract=outcome_contract,
        distribution_destination_id=destination.destination_id,
        eligible_model_routes=(provider.provider_id,),
        per_run_max_tokens=2_000,
        per_run_max_cost_usd=None,
        cumulative_max_tokens=20_000,
        cumulative_max_cost_usd=None,
        cumulative_max_attempts=10,
        cumulative_max_occurrences=10,
        maximum_consecutive_failures=3,
        expires_at=NOW + timedelta(days=30),
    )
    created = await agent.create_routine(await agent.propose_routine(draft))
    seen = {
        item.delivery_id
        for item in await agent.list_deliveries(conversation_id=conversation_id)
    }
    await agent.run_routine_now(created.routine_id, expected_revision=created.revision)
    loop = asyncio.get_running_loop()
    deadline = loop.time() + 15.0
    while loop.time() < deadline:
        deliveries = await agent.list_deliveries(conversation_id=conversation_id)
        # The fixed clock gives every delivery the same time, so match by identity.
        new = [item for item in deliveries if item.delivery_id not in seen]
        if new:
            (delivery,) = new
            return delivery
        await asyncio.sleep(0.01)
    raise AssertionError(await agent.inspect_routine(created.routine_id))


def _columns(
    board: WorkBoard,
) -> dict[WorkColumn, set[tuple[WorkKind, str, str | None]]]:
    return {
        column: {(card.kind, card.id, card.delivery_id) for card in cards}
        for column, cards in board.columns.items()
    }


async def test_work_board_shows_real_jobs_routines_and_results_until_dismissed(
    tmp_path: Path,
) -> None:
    database = tmp_path / "values.sqlite"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE current_value (value INTEGER NOT NULL)")
        connection.execute("INSERT INTO current_value VALUES (7)")
    provider = MockModelProvider(
        (
            _stop("Foreground authorization complete."),
            _stop("The current value is 7."),
            _stop("No export was produced."),
        ),
        provider_id="mock:work-board",
        complete_pricing=True,
    )
    options = {
        "root": tmp_path,
        "workspace": workspace_for(tmp_path),
        "model": provider,
        "model_profile": _profile(provider),
        "clock": lambda: NOW,
    }
    created = await Agent.create(NAME, **options)  # type: ignore[arg-type]
    agent_id = created.id
    await created.close()
    state_path = tmp_path / "agents" / NAME / "state.db"
    assert state_path.is_file()
    await _seed_queued_and_blocked_jobs(state_path, agent_id)

    agent = await Agent.open(NAME, **options)  # type: ignore[arg-type]
    try:
        source = await agent.attach(SQLiteSource(database))
        (resource,) = await agent.list_catalog_resources(source_id=source.id)
        origin = await agent.run("Authorize the scheduled value reports.")
        assert origin.conversation_id is not None
        routine_options = {
            "origin_run_id": origin.run_id,
            "conversation_id": origin.conversation_id,
            "source_id": source.id,
            "resource_id": resource.id,
            "provider": provider,
        }
        succeeded = await _run_routine(
            agent,
            title="Current value report",
            capability_id="catalog.describe",
            outcome_contract=no_artifact_outcome_contract(),
            **routine_options,  # type: ignore[arg-type]
        )
        failed = await _run_routine(
            agent,
            title="Current value export",
            capability_id=DATA_EXPORT_TABULAR_CAPABILITY_ID,
            outcome_contract=_csv_export_contract(),
            **routine_options,  # type: ignore[arg-type]
        )
        provider.assert_consumed()
        assert succeeded.conclusion_state is OutcomeState.SUCCEEDED
        assert failed.conclusion_state is OutcomeState.FAILED
        routines = {item.title: item for item in await agent.list_routines()}

        board = await agent.work_board()
        assert board.generated_at == NOW
        assert _columns(board) == {
            WorkColumn.SCHEDULED: {
                (WorkKind.JOB, "job-queued", None),
                (WorkKind.ROUTINE, routines["Current value report"].routine_id, None),
                (WorkKind.ROUTINE, routines["Current value export"].routine_id, None),
            },
            WorkColumn.RUNNING: set(),
            WorkColumn.NEEDS_YOU: {
                (WorkKind.JOB, "job-blocked", None),
                (WorkKind.ROUTINE_RUN, failed.subject_id, failed.delivery_id),
            },
            WorkColumn.DONE: {
                (WorkKind.ROUTINE_RUN, succeeded.subject_id, succeeded.delivery_id),
            },
        }
        titles = {
            card.id: card.title for cards in board.columns.values() for card in cards
        }
        assert titles[succeeded.subject_id] == "Current value report"
        assert titles[failed.subject_id] == "Current value export"
        assert all(
            card.due_at == NOW + timedelta(hours=1)
            for card in board.columns[WorkColumn.SCHEDULED]
            if card.kind is WorkKind.ROUTINE
        )

        assert await agent.acknowledge_delivery(succeeded.delivery_id) is not None
        dismissed = await agent.work_board()
        assert dismissed.columns[WorkColumn.DONE] == ()
        assert (
            dismissed.columns[WorkColumn.NEEDS_YOU]
            == board.columns[WorkColumn.NEEDS_YOU]
        )
    finally:
        await agent.close()


async def test_work_board_rejects_hosted_callers_and_out_of_range_windows(
    tmp_path: Path,
) -> None:
    provider = MockModelProvider((), provider_id="mock:work-board-bounds")
    local = await Agent.create(
        "work-board-bounds",
        root=tmp_path / "local",
        workspace=workspace_for(tmp_path),
        model=provider,
        model_profile=_profile(provider),
        clock=lambda: NOW,
    )
    try:
        assert (await local.work_board(done_window=timedelta(days=7))).generated_at == (
            NOW
        )
        for window in (
            timedelta(0),
            timedelta(days=7, microseconds=1),
            24,
        ):
            with pytest.raises(ValueError, match="done window"):
                await local.work_board(done_window=window)  # type: ignore[arg-type]
    finally:
        await local.close()

    hosted = await Agent.create(
        "work-board-hosted",
        root=tmp_path / "hosted",
        hosted=True,
        model=provider,
        model_profile=_profile(provider),
    )
    try:
        with pytest.raises(ValueError, match="hosted work board is not available yet"):
            await hosted.work_board()
    finally:
        await hosted.close()
