from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone

import pytest

from daita.loop.models import LoopBudgets
from daita.monitors import (
    IntervalSchedule,
    Monitor,
    MonitorCheckpoint,
    MonitorDefinition,
    MonitorFindingSeverity,
    MonitorInspection,
    MonitorOccurrence,
    MonitorOccurrenceKind,
    MonitorOccurrenceClaim,
    MonitorOutcomeCommit,
    MonitorOutcomeProjection,
    MonitorRun,
    MonitorRunStatus,
    MonitorScheduleState,
    MonitorScheduler,
    MonitorSchedulerContractError,
    MonitorScope,
    MonitorStatus,
    MonitorTickLease,
    MonitorTickClaimConflictError,
    MonitorTimingPolicy,
    MonitorVersion,
    monitor_occurrence_id,
    monitor_occurrence_key,
    monitor_run_id,
    monitor_trigger_id,
)
from daita.monitors.store import (
    MonitorClaimResult,
    MonitorOutcomeResult,
)
from daita.operations.models import (
    AgentTrigger,
    Evidence,
    Operation,
    OperationStatus,
    TriggerKind,
)

AGENT_ID = "agent-atlas"
MONITOR_ID = "monitor-backlog"
NOW = datetime(2026, 7, 19, 12, 0, tzinfo=timezone.utc)


def _definition() -> MonitorDefinition:
    return MonitorDefinition(
        name="Order backlog",
        objective="Inspect the current order backlog and cite accepted evidence.",
        scope=MonitorScope(source_ids=("source-orders",)),
        schedule=IntervalSchedule(interval_seconds=60, anchor_at=NOW),
        timing=MonitorTimingPolicy(
            cooldown_seconds=30,
            initial_backoff_seconds=5,
            max_backoff_seconds=20,
        ),
        operation_template={"domain": "data"},
    )


def _inspection() -> MonitorInspection:
    definition = _definition()
    return MonitorInspection(
        monitor=Monitor(
            id=MONITOR_ID,
            agent_id=AGENT_ID,
            status=MonitorStatus.ENABLED,
            current_version=1,
            revision=1,
            created_at=NOW - timedelta(minutes=1),
            updated_at=NOW - timedelta(minutes=1),
        ),
        versions=(
            MonitorVersion(
                id="monitor-version-1",
                agent_id=AGENT_ID,
                monitor_id=MONITOR_ID,
                version=1,
                definition=definition,
                content_hash=definition.content_hash,
                proposal_id="monitor-proposal-1",
                created_at=NOW - timedelta(minutes=1),
            ),
        ),
        lifecycle=(),
        schedule_state=MonitorScheduleState(
            agent_id=AGENT_ID,
            monitor_id=MONITOR_ID,
            revision=1,
            next_scheduled_at=NOW,
            updated_at=NOW - timedelta(minutes=1),
        ),
    )


class FakeMonitorStore:
    def __init__(self, inspection: MonitorInspection, *, conflict: bool = False):
        self.inspection = inspection
        self.conflict = conflict
        self.claims: list[MonitorOccurrenceClaim | MonitorClaimResult] = []
        self.outcomes: list[MonitorOutcomeCommit] = []

    async def list_due_monitors(self, agent_id, *, now, limit):
        assert agent_id == AGENT_ID
        state = self.inspection.schedule_state
        return (
            (self.inspection,)
            if state.next_scheduled_at is not None and state.next_scheduled_at <= now
            else ()
        )

    async def inspect_monitor(self, agent_id, monitor_id):
        assert agent_id == AGENT_ID
        return self.inspection if monitor_id == MONITOR_ID else None

    async def load_occurrence_by_trigger(self, agent_id, trigger_id):
        assert agent_id == AGENT_ID
        return next(
            (
                occurrence
                for occurrence in self.inspection.occurrences
                if occurrence.trigger_id == trigger_id
            ),
            None,
        )

    async def claim_monitor_occurrence(
        self,
        claim,
        *,
        expected_monitor_revision,
        expected_schedule_revision,
        checked_at,
    ):
        self.claims.append(claim)
        if self.conflict:
            raise MonitorTickClaimConflictError(
                claim.occurrence.id,
                holder_id="other-scheduler",
                fencing_token=1,
                expires_at=checked_at + timedelta(seconds=30),
            )
        occurrences = {
            item.id: item for item in (*self.inspection.occurrences, claim.occurrence)
        }
        runs = {item.id: item for item in (*self.inspection.runs, claim.run)}
        self.inspection = replace(
            self.inspection,
            occurrences=tuple(occurrences.values()),
            leases=(*self.inspection.leases, claim.lease),
            runs=tuple(runs.values()),
        )
        return MonitorClaimResult(claim.occurrence, claim.lease, claim.run)

    async def commit_monitor_outcome(
        self,
        commit,
        *,
        expected_monitor_revision,
        expected_schedule_revision,
        checked_at,
    ):
        self.outcomes.append(commit)
        occurrence = next(
            claim.occurrence
            for claim in reversed(self.claims)
            if claim.occurrence.id == commit.guard.occurrence_id
        )
        occurrences = {
            item.id: item for item in (*self.inspection.occurrences, occurrence)
        }
        runs = {item.id: item for item in (*self.inspection.runs, commit.run)}
        leases = tuple(
            commit.released_lease if item.id == commit.released_lease.id else item
            for item in self.inspection.leases
        )
        self.inspection = replace(
            self.inspection,
            schedule_state=commit.schedule_state,
            occurrences=tuple(occurrences.values()),
            leases=leases,
            runs=tuple(runs.values()),
            findings=(
                self.inspection.findings
                if commit.finding is None
                else (*self.inspection.findings, commit.finding)
            ),
            checkpoints=(
                self.inspection.checkpoints
                if commit.checkpoint is None
                else (*self.inspection.checkpoints, commit.checkpoint)
            ),
        )
        return MonitorOutcomeResult(self.inspection, commit.run, commit.finding)


@dataclass
class FakeSnapshot:
    trigger: AgentTrigger
    operation: Operation
    evidence: tuple[Evidence, ...]
    budgets: LoopBudgets


@dataclass
class FakeVersionedOperation:
    snapshot: FakeSnapshot


class FakeOperationStore:
    def __init__(self):
        self.by_trigger: dict[str, FakeVersionedOperation] = {}

    async def load_by_trigger(self, trigger_id):
        return self.by_trigger.get(trigger_id)


class FakeRunner:
    def __init__(
        self,
        operations: FakeOperationStore,
        statuses: tuple[OperationStatus, ...],
        *,
        accepted: bool = True,
    ):
        self.operations = operations
        self.statuses = list(statuses)
        self.accepted = accepted
        self.triggers: list[AgentTrigger] = []

    async def run(self, trigger: AgentTrigger):
        self.triggers.append(trigger)
        status = self.statuses.pop(0)
        existing = self.operations.by_trigger.get(trigger.id)
        operation_id = (
            "operation-monitor-1"
            if existing is None
            else existing.snapshot.operation.id
        )
        terminal = status in {
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
            OperationStatus.INTERRUPTED,
        }
        operation = Operation(
            id=operation_id,
            agent_id=AGENT_ID,
            trigger_id=trigger.id,
            status=status,
            created_at=trigger.created_at,
            updated_at=NOW + timedelta(seconds=len(self.triggers)),
            final_text=(
                "Backlog is high [evidence:evidence-monitor-1]."
                if status is OperationStatus.SUCCEEDED
                else None
            ),
            terminal_reason="completed" if terminal else None,
        )
        evidence: tuple[Evidence, ...] = ()
        if status is OperationStatus.SUCCEEDED:
            evidence = (
                Evidence(
                    id="evidence-monitor-1",
                    operation_id=operation_id,
                    task_id="task-monitor-1",
                    turn_id="turn-monitor-1",
                    capability_id="data.read",
                    executor_id="data.read.executor",
                    kind="query.result",
                    schema_version=1,
                    attempt=1,
                    accepted=self.accepted,
                    payload={"backlog": 12},
                    content_hash="sha256:" + "a" * 64,
                    created_at=NOW + timedelta(seconds=1),
                ),
            )
        self.operations.by_trigger[trigger.id] = FakeVersionedOperation(
            FakeSnapshot(trigger, operation, evidence, LoopBudgets())
        )


class CancellingOnceRunner(FakeRunner):
    async def run(self, trigger: AgentTrigger):
        if not self.triggers:
            self.triggers.append(trigger)
            raise asyncio.CancelledError
        return await super().run(trigger)


class FakeProjector:
    def __init__(self, projection: MonitorOutcomeProjection):
        self.projection = projection
        self.calls: list[
            tuple[MonitorDefinition, FakeSnapshot, MonitorCheckpoint | None]
        ] = []

    async def project(self, *, definition, operation, checkpoint):
        self.calls.append((definition, operation, checkpoint))
        return self.projection


class IdFactory:
    def __init__(self):
        self.value = 0

    def __call__(self, prefix):
        self.value += 1
        return f"{prefix}-{self.value}"


def _scheduler(store, operations, runner, projector, *, clock):
    return MonitorScheduler(
        agent_id=AGENT_ID,
        store=store,
        operations=operations,
        runner=runner,
        projector=projector,
        holder_id="scheduler-1",
        lease_seconds=30,
        clock=clock,
        id_factory=IdFactory(),
    )


def _manual_claim(store: FakeMonitorStore) -> MonitorClaimResult:
    key = monitor_occurrence_key(
        agent_id=AGENT_ID,
        monitor_id=MONITOR_ID,
        monitor_version=1,
        kind=MonitorOccurrenceKind.RUN_NOW,
        scheduled_for=NOW,
        manual_key="manual-run-1",
    )
    occurrence = MonitorOccurrence(
        id=monitor_occurrence_id(key),
        agent_id=AGENT_ID,
        monitor_id=MONITOR_ID,
        monitor_version=1,
        kind=MonitorOccurrenceKind.RUN_NOW,
        scheduled_for=NOW,
        occurrence_key=key,
        trigger_id=monitor_trigger_id(key),
        run_id=monitor_run_id(key),
        created_at=NOW,
        manual_key="manual-run-1",
    )
    lease = MonitorTickLease(
        id="monitor-lease-manual-1",
        agent_id=AGENT_ID,
        monitor_id=MONITOR_ID,
        occurrence_id=occurrence.id,
        holder_id="scheduler-1",
        fencing_token=1,
        claimed_at=NOW,
        expires_at=NOW + timedelta(seconds=30),
    )
    run = MonitorRun(
        id=occurrence.run_id,
        agent_id=AGENT_ID,
        monitor_id=MONITOR_ID,
        occurrence_id=occurrence.id,
        trigger_id=occurrence.trigger_id,
        attempt=1,
        fencing_token=1,
        status=MonitorRunStatus.PENDING,
        started_at=NOW,
    )
    store.inspection = replace(
        store.inspection,
        occurrences=(occurrence,),
        leases=(lease,),
        runs=(run,),
    )
    result = MonitorClaimResult(occurrence, lease, run)
    store.claims.append(result)
    return result


async def test_due_match_uses_one_stable_normal_trigger_and_atomic_outcome():
    store = FakeMonitorStore(_inspection())
    operations = FakeOperationStore()
    runner = FakeRunner(operations, (OperationStatus.SUCCEEDED,))
    projector = FakeProjector(
        MonitorOutcomeProjection(
            matched=True,
            evidence_id="evidence-monitor-1",
            cursor={"last_order_id": 41},
            severity=MonitorFindingSeverity.WARNING,
            summary="Order backlog exceeded the threshold.",
            details={"backlog": 12},
            dedupe_key="backlog-high",
        )
    )
    scheduler = _scheduler(
        store,
        operations,
        runner,
        projector,
        clock=lambda: NOW + timedelta(seconds=5),
    )

    result = await scheduler.run_due(NOW + timedelta(seconds=1))

    assert result[0].run_status is MonitorRunStatus.SUCCEEDED
    assert len(runner.triggers) == 1
    trigger = runner.triggers[0]
    assert trigger.kind is TriggerKind.MONITOR
    assert trigger.source_id == MONITOR_ID
    assert trigger.created_at == NOW
    assert trigger.payload["message"] == _definition().objective
    assert "fencing_token" not in trigger.payload
    outcome = store.outcomes[0]
    assert outcome.checkpoint is not None
    assert dict(outcome.checkpoint.cursor) == {"last_order_id": 41}
    assert outcome.finding is not None
    assert outcome.finding.evidence_id == "evidence-monitor-1"
    assert outcome.schedule_state.next_scheduled_at == NOW + timedelta(minutes=1)
    assert outcome.schedule_state.cooldown_until == NOW + timedelta(seconds=35)
    assert all(event.operation_id is None for event in outcome.events)
    assert outcome.events[-1].payload["evidence_id"] == "evidence-monitor-1"


async def test_claim_conflict_skips_without_running_an_operation():
    store = FakeMonitorStore(_inspection(), conflict=True)
    operations = FakeOperationStore()
    runner = FakeRunner(operations, (OperationStatus.SUCCEEDED,))
    scheduler = _scheduler(
        store,
        operations,
        runner,
        FakeProjector(MonitorOutcomeProjection(matched=False)),
        clock=lambda: NOW + timedelta(seconds=1),
    )

    result = await scheduler.run_due(NOW)

    assert result[0].claimed is False
    assert result[0].reason == "claim_conflict"
    assert runner.triggers == []
    assert store.outcomes == []


async def test_waiting_reclaim_reuses_trigger_and_operation_with_higher_fence():
    store = FakeMonitorStore(_inspection())
    operations = FakeOperationStore()
    runner = FakeRunner(
        operations,
        (OperationStatus.WAITING_FOR_INPUT, OperationStatus.SUCCEEDED),
    )
    projector = FakeProjector(MonitorOutcomeProjection(matched=False))
    current = NOW + timedelta(seconds=2)
    scheduler = _scheduler(
        store,
        operations,
        runner,
        projector,
        clock=lambda: current,
    )

    first = await scheduler.run_due(NOW)
    current = NOW + timedelta(seconds=4)
    second = await scheduler.run_due(NOW + timedelta(seconds=3))

    assert first[0].run_status is MonitorRunStatus.WAITING
    assert second[0].run_status is MonitorRunStatus.SUCCEEDED
    assert runner.triggers[0] == runner.triggers[1]
    assert store.claims[0].occurrence == store.claims[1].occurrence
    assert store.claims[1].lease.fencing_token == 2
    assert store.claims[1].run.attempt == 2
    assert store.claims[1].run.operation_id == "operation-monitor-1"
    assert store.outcomes[0].schedule_state.next_scheduled_at == NOW
    assert store.outcomes[1].schedule_state.next_scheduled_at == NOW + timedelta(
        minutes=1
    )


async def test_unaccepted_projection_evidence_cannot_commit_monitor_outcome():
    store = FakeMonitorStore(_inspection())
    operations = FakeOperationStore()
    runner = FakeRunner(
        operations,
        (OperationStatus.SUCCEEDED,),
        accepted=False,
    )
    projector = FakeProjector(
        MonitorOutcomeProjection(
            matched=True,
            evidence_id="evidence-monitor-1",
            summary="Invalid finding.",
            dedupe_key="invalid-finding",
        )
    )
    scheduler = _scheduler(
        store,
        operations,
        runner,
        projector,
        clock=lambda: NOW + timedelta(seconds=2),
    )

    with pytest.raises(MonitorSchedulerContractError, match="not accepted"):
        await scheduler.run_due(NOW)

    assert len(store.claims) == 1
    assert store.outcomes == []


async def test_failure_applies_backoff_and_catches_up_once_after_downtime():
    store = FakeMonitorStore(_inspection())
    operations = FakeOperationStore()
    runner = FakeRunner(operations, (OperationStatus.FAILED,))
    projector = FakeProjector(MonitorOutcomeProjection(matched=False))
    finished_at = NOW + timedelta(minutes=3, seconds=5)
    scheduler = _scheduler(
        store,
        operations,
        runner,
        projector,
        clock=lambda: finished_at,
    )

    result = await scheduler.run_due(NOW + timedelta(minutes=3))

    assert result[0].run_status is MonitorRunStatus.FAILED
    outcome = store.outcomes[0]
    assert outcome.schedule_state.consecutive_failures == 1
    assert outcome.schedule_state.backoff_until == finished_at + timedelta(seconds=5)
    assert outcome.schedule_state.next_scheduled_at == NOW + timedelta(minutes=4)
    assert outcome.checkpoint is None
    assert outcome.finding is None
    assert projector.calls == []


async def test_cancellation_leaves_claim_for_expired_deterministic_reclaim():
    store = FakeMonitorStore(_inspection())
    operations = FakeOperationStore()
    runner = CancellingOnceRunner(operations, (OperationStatus.SUCCEEDED,))
    projector = FakeProjector(MonitorOutcomeProjection(matched=False))
    current = NOW + timedelta(seconds=1)
    scheduler = _scheduler(
        store,
        operations,
        runner,
        projector,
        clock=lambda: current,
    )

    with pytest.raises(asyncio.CancelledError):
        await scheduler.run_due(NOW)
    assert store.outcomes == []

    current = NOW + timedelta(seconds=35)
    result = await scheduler.run_due(NOW + timedelta(seconds=31))

    assert result[0].run_status is MonitorRunStatus.SUCCEEDED
    assert runner.triggers[0] == runner.triggers[1]
    assert store.claims[0].occurrence == store.claims[1].occurrence
    assert store.claims[1].lease.fencing_token == 2
    assert store.claims[1].run.attempt == 2
    assert len(store.outcomes) == 1


async def test_run_now_claim_uses_normal_trigger_without_moving_schedule():
    store = FakeMonitorStore(_inspection())
    claim = _manual_claim(store)
    operations = FakeOperationStore()
    runner = FakeRunner(operations, (OperationStatus.SUCCEEDED,))
    scheduler = _scheduler(
        store,
        operations,
        runner,
        FakeProjector(MonitorOutcomeProjection(matched=False)),
        clock=lambda: NOW + timedelta(seconds=5),
    )

    result = await scheduler.run_claimed(claim)

    assert result.run_status is MonitorRunStatus.SUCCEEDED
    assert runner.triggers[0].id == claim.occurrence.trigger_id
    assert runner.triggers[0].kind is TriggerKind.MONITOR
    assert store.outcomes[0].schedule_state.next_scheduled_at == NOW


async def test_waiting_run_now_reclaims_same_trigger_after_approval_wake():
    store = FakeMonitorStore(_inspection())
    claim = _manual_claim(store)
    operations = FakeOperationStore()
    runner = FakeRunner(
        operations,
        (OperationStatus.WAITING_FOR_APPROVAL, OperationStatus.SUCCEEDED),
    )
    current = NOW + timedelta(seconds=2)
    scheduler = _scheduler(
        store,
        operations,
        runner,
        FakeProjector(MonitorOutcomeProjection(matched=False)),
        clock=lambda: current,
    )

    first = await scheduler.run_claimed(claim)
    current = NOW + timedelta(seconds=4)
    second = await scheduler.resume_trigger(claim.occurrence.trigger_id)

    assert first.run_status is MonitorRunStatus.WAITING
    assert second.run_status is MonitorRunStatus.SUCCEEDED
    assert runner.triggers[0] == runner.triggers[1]
    assert store.claims[-1].lease.fencing_token == 2
    assert store.claims[-1].run.attempt == 2
    assert store.claims[-1].run.operation_id == "operation-monitor-1"
