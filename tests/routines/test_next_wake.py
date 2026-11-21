from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import cast

from daita import Agent, IntervalSchedule, RoutineIntent, SQLiteSource
from daita.artifacts.store import AgentHomeArtifactStore
from daita.capability_runtime import CapabilityRuntime
from daita.distribution import DistributionOwner
from daita.loop.models import LoopExit, RunInput
from daita.routines.models import RoutineOccurrence, RoutineState
from daita.routines.owner import RoutineOwner
from daita.routines.supervisor import RoutineSupervisor
from daita.storage.sqlite import SQLiteStateStore
from tests.routines._supervisor_support import _ids
from tests.routines.test_supervisor import (
    _SCOPE,
    NOW,
    _Owner,
    _routine,
    _seed_conversation,
    _UnusedRuntime,
)
from tests.support.jobs import JobModel
from tests.support.workspace import workspace_for


class _Clock:
    def __init__(self, now: datetime) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


async def _supervisor(tmp_path, store, clock, execute=None) -> RoutineSupervisor:
    return RoutineSupervisor(
        agent_id="agent-1",
        store=store,
        owner=cast(RoutineOwner, _Owner()),
        runtime=cast(CapabilityRuntime, _UnusedRuntime()),
        distribution=DistributionOwner(agent_id="agent-1", store=store),
        artifacts=await AgentHomeArtifactStore.open(
            agent_id="agent-1", agent_home=tmp_path, registry=store
        ),
        execute_run=execute or _never,
        machine_scope=_SCOPE,
        model_routes=("mock",),
        per_run_max_tokens=1_000,
        per_run_max_cost_usd=Decimal("0.10"),
        clock=clock,
        id_factory=_ids(),
        poll_seconds=0.02,
    )


async def _never(occurrence: RoutineOccurrence, run: RunInput) -> LoopExit | None:
    raise AssertionError("not expected to run")


async def test_wake_follows_active_routine_pause_and_revision(tmp_path) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        await _seed_conversation(store)
        supervisor = await _supervisor(tmp_path, store, _Clock(NOW))
        assert await supervisor.next_wake_at() is None
        routine = await store.admit_scheduled_routine(_routine())
        assert routine.next_due_at is not None
        wake = await supervisor.next_wake_at()
        assert wake == routine.next_due_at and wake.utcoffset() == timedelta(0)

        later = replace(
            routine,
            schedule=IntervalSchedule(7200, NOW + timedelta(days=1)),
            revision=routine.revision + 1,
        )
        revised = await store.revise_scheduled_routine(
            later, expected_revision=routine.revision
        )
        assert revised is not None and revised.next_due_at != routine.next_due_at
        assert await supervisor.next_wake_at() == revised.next_due_at

        paused = await store.transition_scheduled_routine(
            "agent-1",
            routine.routine_id,
            expected_revision=revised.revision,
            state=RoutineState.PAUSED,
            transitioned_at=NOW,
        )
        assert paused is not None
        assert await supervisor.next_wake_at() is None
    finally:
        await store.close()


async def test_wake_waits_for_quota_hold_and_stale_claim_lease(tmp_path) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    try:
        await _seed_conversation(store)
        routine = await store.admit_scheduled_routine(_routine())
        assert routine.next_due_at is not None
        due = routine.next_due_at
        clock = _Clock(due + timedelta(minutes=10))
        supervisor = await _supervisor(tmp_path, store, clock)
        assert await supervisor.next_wake_at() == due

        hold_end = clock.now + timedelta(minutes=5)
        supervisor._quota_holds["mock"] = (hold_end, timedelta(minutes=5))
        assert await supervisor.next_wake_at() == hold_end
        clock.now = hold_end
        assert await supervisor.next_wake_at() == due  # hold over: slot is just due

        occurrence = await store.claim_due_routine_occurrence(
            "agent-1",
            routine.routine_id,
            expected_revision=routine.revision,
            expected_due_at=due,
            claimed_at=due,
            claim_token="claim-1",
        )
        assert occurrence is not None and occurrence.lease_expires_at is not None
        lease_end = occurrence.lease_expires_at
        # The claimed slot is no longer claimable; only recovery has work left.
        assert await supervisor.next_wake_at() == lease_end
        tick = timedelta(microseconds=1)
        assert not await store.recover_stale_routine_occurrences(
            "agent-1",
            recovered_at=lease_end - tick,
            claim_token_factory=lambda _: "claim-2",
        )
        assert await store.recover_stale_routine_occurrences(
            "agent-1", recovered_at=lease_end, claim_token_factory=lambda _: "claim-2"
        )
    finally:
        await store.close()


async def test_supervisor_runs_work_at_the_returned_instant(tmp_path) -> None:
    store = await SQLiteStateStore.open(tmp_path / "state.db")
    ran = asyncio.Event()

    async def execute(occurrence: RoutineOccurrence, run: RunInput) -> None:
        ran.set()

    supervisor = None
    try:
        await _seed_conversation(store)
        await store.admit_scheduled_routine(_routine())
        clock = _Clock(NOW - timedelta(hours=1))
        supervisor = await _supervisor(tmp_path, store, clock, execute)
        wake = await supervisor.next_wake_at()
        assert wake is not None and wake > clock.now
        await supervisor.start()
        clock.now = wake - timedelta(microseconds=1)
        supervisor.wake()
        await asyncio.sleep(0.1)
        assert not ran.is_set()
        clock.now = wake
        supervisor.wake()
        await asyncio.wait_for(ran.wait(), 5)
    finally:
        if supervisor is not None:
            await supervisor.close()
        await store.close()


async def test_agent_wake_ignores_catalog_and_tracks_routines(tmp_path) -> None:
    database = tmp_path / "source.db"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE current_value (value INTEGER)")
    agent = await Agent.create(
        "next-wake",
        root=tmp_path,
        workspace=workspace_for(tmp_path),
        model=JobModel(),
        model_profile=JobModel.model_profile,
    )
    try:
        assert await agent.next_wake_at() is None
        await agent.attach(SQLiteSource(database, name="Current"))
        assert await agent.next_wake_at() is None  # catalog upkeep never wakes

        anchor = datetime.now(UTC) + timedelta(hours=1)
        created = await agent.create_routine(
            RoutineIntent(
                title="Report",
                prompt="Report the current value.",
                schedule=IntervalSchedule(3_600, anchor),
                sources=("Current",),
            )
        )
        assert created.next_due_at is not None
        assert await agent.next_wake_at() == created.next_due_at
        paused = await agent.pause_routine(
            created.routine_id, expected_revision=created.revision
        )
        assert await agent.next_wake_at() is None
        await agent.resume_routine(paused.routine_id, expected_revision=paused.revision)
        assert await agent.next_wake_at() is not None
    finally:
        await agent.close()
