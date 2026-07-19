from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita.events import RuntimeEvent
from daita.identity import AgentIdentity
from daita.monitors import (
    IntervalSchedule,
    MonitorCheckpoint,
    MonitorDefinition,
    MonitorOccurrence,
    MonitorOccurrenceClaim,
    MonitorOccurrenceKind,
    MonitorOutcomeCommit,
    MonitorRun,
    MonitorRunStatus,
    MonitorScope,
    MonitorService,
    MonitorStatus,
    MonitorTickClaimConflictError,
    MonitorTickLease,
    MonitorTickLeaseGuard,
    StaleMonitorFenceError,
    monitor_occurrence_id,
    monitor_occurrence_key,
    monitor_run_id,
    monitor_trigger_id,
)
from daita.storage.sqlite import SQLiteOperationStore

UTC = timezone.utc
NOW = datetime(2026, 7, 19, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class Ids:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}-{self.count}"


def _definition() -> MonitorDefinition:
    return MonitorDefinition(
        name="Orders backlog",
        objective="Inspect orders and record a durable threshold finding.",
        scope=MonitorScope(
            source_ids=("source-orders",),
            resource_ids=("resource-orders",),
        ),
        schedule=IntervalSchedule(interval_seconds=300, anchor_at=NOW),
    )


async def _open(path: Path) -> SQLiteOperationStore:
    store = await SQLiteOperationStore.open(path)
    await store.initialize_identity(
        AgentIdentity(id="agent-1", display_name="Agent 1", created_at=NOW)
    )
    return store


async def _activated(
    store: SQLiteOperationStore,
    clock: Clock,
    ids: Ids,
):
    service = MonitorService(
        agent_id="agent-1",
        store=store,
        clock=clock,
        id_factory=ids,
    )
    proposal = await service.propose(
        "orders-backlog",
        _definition(),
        idempotency_key="create-orders",
    )
    inspection = await service.confirm(
        proposal.id,
        candidate_hash=proposal.candidate_hash,
        actor_id="user-1",
        reason="Schedule, scope, and objective confirmed.",
    )
    return service, proposal, inspection


async def test_lifecycle_events_and_run_now_survive_reopen(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    clock = Clock()
    ids = Ids()
    store = await _open(path)
    service, proposal, activated = await _activated(store, clock, ids)

    replay = await service.propose(
        "orders-backlog",
        _definition(),
        idempotency_key="create-orders",
    )
    assert replay == proposal
    assert await store.list_due_monitors(
        "agent-1",
        now=NOW + timedelta(minutes=5),
        limit=10,
    ) == (activated,)

    clock.value += timedelta(minutes=1)
    paused = await service.pause(
        activated.monitor.id,
        actor_id="user-1",
        reason="Maintenance.",
        idempotency_key="pause-orders",
    )
    paused_replay = await service.pause(
        activated.monitor.id,
        actor_id="user-1",
        reason="Maintenance.",
        idempotency_key="pause-orders",
    )
    assert paused_replay.monitor == paused.monitor

    first_manual = await service.run_now(
        activated.monitor.id,
        idempotency_key="manual-orders-1",
        holder_id="host-1",
        lease_seconds=60,
    )
    clock.value += timedelta(seconds=5)
    replayed_manual = await service.run_now(
        activated.monitor.id,
        idempotency_key="manual-orders-1",
        holder_id="host-1",
        lease_seconds=60,
    )
    assert replayed_manual == first_manual

    events = await store.read_after("agent-1", None, limit=100)
    assert [event.event.type for event in events] == [
        "monitor.proposed",
        "monitor.activated",
        "monitor.pause",
        "monitor.claimed",
    ]
    assert {event.event.monitor_id for event in events} == {"orders-backlog"}
    await store.close()

    reopened = await _open(path)
    inspection = await reopened.inspect_monitor("agent-1", "orders-backlog")
    assert inspection is not None
    assert inspection.monitor.status is MonitorStatus.PAUSED
    assert inspection.proposals == (proposal,)
    assert inspection.occurrences == (first_manual.occurrence,)
    assert inspection.runs == (first_manual.run,)
    assert inspection.leases == (first_manual.lease,)
    assert (
        await reopened.load_occurrence_by_trigger(
            "agent-1",
            first_manual.occurrence.trigger_id,
        )
        == first_manual.occurrence
    )
    await reopened.close()


async def test_waiting_run_now_can_reclaim_with_a_higher_fence(tmp_path: Path) -> None:
    store = await _open(tmp_path / "state.db")
    clock = Clock()
    service, _, inspection = await _activated(store, clock, Ids())
    first = await service.run_now(
        inspection.monitor.id,
        idempotency_key="manual-waiting-1",
        holder_id="host-1",
        lease_seconds=60,
    )
    waiting_at = NOW + timedelta(seconds=5)
    waiting_run = replace(first.run, status=MonitorRunStatus.WAITING)
    released = replace(
        first.lease,
        released_at=waiting_at,
        release_reason="waiting",
    )
    waiting_state = replace(
        inspection.schedule_state,
        revision=2,
        updated_at=waiting_at,
        last_occurrence_id=first.occurrence.id,
        last_run_id=first.run.id,
    )
    await store.commit_monitor_outcome(
        MonitorOutcomeCommit(
            guard=MonitorTickLeaseGuard(
                agent_id="agent-1",
                monitor_id=inspection.monitor.id,
                occurrence_id=first.occurrence.id,
                holder_id=first.lease.holder_id,
                fencing_token=1,
            ),
            run=waiting_run,
            released_lease=released,
            schedule_state=waiting_state,
            events=(
                RuntimeEvent(
                    id="event-manual-waiting",
                    type="monitor.waiting",
                    agent_id="agent-1",
                    operation_id=None,
                    monitor_id=inspection.monitor.id,
                    created_at=waiting_at,
                ),
            ),
        ),
        expected_monitor_revision=1,
        expected_schedule_revision=1,
        checked_at=waiting_at,
    )

    reclaimed_at = waiting_at + timedelta(seconds=1)
    next_lease = MonitorTickLease(
        id="lease-manual-2",
        agent_id="agent-1",
        monitor_id=inspection.monitor.id,
        occurrence_id=first.occurrence.id,
        holder_id="host-1",
        fencing_token=2,
        claimed_at=reclaimed_at,
        expires_at=reclaimed_at + timedelta(seconds=60),
    )
    next_run = replace(
        waiting_run,
        attempt=2,
        fencing_token=2,
        status=MonitorRunStatus.PENDING,
        started_at=reclaimed_at,
    )
    reclaimed = await store.claim_monitor_occurrence(
        MonitorOccurrenceClaim(
            occurrence=first.occurrence,
            lease=next_lease,
            run=next_run,
            event=RuntimeEvent(
                id="event-manual-reclaimed",
                type="monitor.claimed",
                agent_id="agent-1",
                operation_id=None,
                monitor_id=inspection.monitor.id,
                created_at=reclaimed_at,
            ),
        ),
        expected_monitor_revision=1,
        expected_schedule_revision=2,
        checked_at=reclaimed_at,
    )

    assert reclaimed.occurrence == first.occurrence
    assert reclaimed.lease.fencing_token == 2
    assert reclaimed.run.attempt == 2
    await store.close()


async def test_scheduled_claim_race_and_fenced_outcome_are_atomic(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    store = await _open(path)
    _, _, inspection = await _activated(store, Clock(), Ids())
    scheduled_for = inspection.schedule_state.next_scheduled_at
    assert scheduled_for is not None
    key = monitor_occurrence_key(
        agent_id="agent-1",
        monitor_id=inspection.monitor.id,
        monitor_version=1,
        kind=MonitorOccurrenceKind.SCHEDULED,
        scheduled_for=scheduled_for,
    )
    occurrence = MonitorOccurrence(
        id=monitor_occurrence_id(key),
        agent_id="agent-1",
        monitor_id=inspection.monitor.id,
        monitor_version=1,
        kind=MonitorOccurrenceKind.SCHEDULED,
        scheduled_for=scheduled_for,
        occurrence_key=key,
        trigger_id=monitor_trigger_id(key),
        run_id=monitor_run_id(key),
        created_at=scheduled_for,
    )
    lease = MonitorTickLease(
        id="lease-scheduled-1",
        agent_id="agent-1",
        monitor_id=inspection.monitor.id,
        occurrence_id=occurrence.id,
        holder_id="host-1",
        fencing_token=1,
        claimed_at=scheduled_for,
        expires_at=scheduled_for + timedelta(minutes=1),
    )
    run = MonitorRun(
        id=occurrence.run_id,
        agent_id="agent-1",
        monitor_id=inspection.monitor.id,
        occurrence_id=occurrence.id,
        trigger_id=occurrence.trigger_id,
        attempt=1,
        fencing_token=1,
        status=MonitorRunStatus.PENDING,
        started_at=scheduled_for,
    )

    def event(event_id: str, event_type: str) -> RuntimeEvent:
        return RuntimeEvent(
            id=event_id,
            type=event_type,
            agent_id="agent-1",
            operation_id=None,
            monitor_id=inspection.monitor.id,
            created_at=scheduled_for,
        )

    claim = MonitorOccurrenceClaim(
        occurrence=occurrence,
        lease=lease,
        run=run,
        event=event("event-scheduled-claim", "monitor.claimed"),
    )
    claimed = await store.claim_monitor_occurrence(
        claim,
        expected_monitor_revision=1,
        expected_schedule_revision=1,
        checked_at=scheduled_for,
    )
    assert claimed.run == run

    losing_lease = replace(lease, id="lease-loser", holder_id="host-2")
    with pytest.raises(MonitorTickClaimConflictError):
        await store.claim_monitor_occurrence(
            replace(
                claim,
                lease=losing_lease,
                event=event("event-losing-claim", "monitor.claimed"),
            ),
            expected_monitor_revision=1,
            expected_schedule_revision=1,
            checked_at=scheduled_for,
        )

    completed_at = scheduled_for + timedelta(seconds=10)
    completed_run = replace(
        run,
        status=MonitorRunStatus.SKIPPED,
        completed_at=completed_at,
    )
    released = replace(
        lease,
        released_at=completed_at,
        release_reason="completed",
    )
    checkpoint = MonitorCheckpoint(
        id="checkpoint-scheduled-1",
        agent_id="agent-1",
        monitor_id=inspection.monitor.id,
        version=1,
        run_id=run.id,
        cursor={"scheduled_for": scheduled_for.isoformat()},
        cursor_hash=(
            "sha256:" "a9eda201287544fd65d592f319e4da8feabd023e45cb6d9be4a38811f487efa2"
        ),
        created_at=completed_at,
    )
    next_state = replace(
        inspection.schedule_state,
        revision=2,
        next_scheduled_at=scheduled_for + timedelta(minutes=5),
        updated_at=completed_at,
        last_scheduled_at=scheduled_for,
        checkpoint_version=1,
        last_occurrence_id=occurrence.id,
        last_run_id=run.id,
    )
    outcome = MonitorOutcomeCommit(
        guard=MonitorTickLeaseGuard(
            agent_id="agent-1",
            monitor_id=inspection.monitor.id,
            occurrence_id=occurrence.id,
            holder_id="host-1",
            fencing_token=1,
        ),
        run=completed_run,
        released_lease=released,
        schedule_state=next_state,
        checkpoint=checkpoint,
        events=(event("event-scheduled-outcome", "monitor.completed"),),
    )
    committed = await store.commit_monitor_outcome(
        outcome,
        expected_monitor_revision=1,
        expected_schedule_revision=1,
        checked_at=completed_at,
    )
    assert committed.run == completed_run
    assert committed.inspection.checkpoints == (checkpoint,)
    assert committed.inspection.schedule_state == next_state
    assert committed.inspection.leases[-1] == released

    with pytest.raises(StaleMonitorFenceError):
        await store.commit_monitor_outcome(
            outcome,
            expected_monitor_revision=1,
            expected_schedule_revision=2,
            checked_at=completed_at,
        )
    await store.close()
