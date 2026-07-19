from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from daita.events import RuntimeEvent
from daita.monitors import (
    CronSchedule,
    Monitor,
    MonitorCheckpoint,
    MonitorConfirmation,
    MonitorConfirmationCommit,
    MonitorConfirmationDecision,
    MonitorDefinition,
    MonitorFinding,
    MonitorFindingSeverity,
    MonitorLifecycleAction,
    MonitorLifecycleCommit,
    MonitorLifecycleRecord,
    MonitorOccurrence,
    MonitorOccurrenceClaim,
    MonitorOccurrenceKind,
    MonitorOutcomeCommit,
    MonitorRun,
    MonitorRunStatus,
    MonitorScheduleState,
    MonitorScope,
    MonitorStatus,
    MonitorTickLease,
    MonitorTickLeaseGuard,
    MonitorVersion,
    monitor_occurrence_id,
    monitor_occurrence_key,
    monitor_run_id,
    monitor_trigger_id,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _event(
    event_id: str, event_type: str, *, monitor_id: str = "monitor-a"
) -> RuntimeEvent:
    return RuntimeEvent(
        id=event_id,
        type=event_type,
        agent_id="agent-a",
        operation_id=None,
        monitor_id=monitor_id,
        created_at=NOW,
    )


def _activation() -> tuple[
    Monitor,
    MonitorVersion,
    MonitorLifecycleRecord,
    MonitorScheduleState,
    MonitorConfirmation,
]:
    definition = MonitorDefinition(
        name="Daily monitor",
        objective="Inspect the selected resource for quality regressions.",
        scope=MonitorScope(source_ids=("source-a",), resource_ids=("resource-a",)),
        schedule=CronSchedule("0 8 * * *"),
    )
    monitor = Monitor(
        id="monitor-a",
        agent_id="agent-a",
        status=MonitorStatus.ENABLED,
        current_version=1,
        revision=1,
        created_at=NOW,
        updated_at=NOW,
    )
    version = MonitorVersion(
        id="monitor-version-a",
        agent_id="agent-a",
        monitor_id=monitor.id,
        version=1,
        definition=definition,
        content_hash=definition.content_hash,
        proposal_id="proposal-a",
        created_at=NOW,
    )
    lifecycle = MonitorLifecycleRecord(
        id="lifecycle-a",
        agent_id="agent-a",
        monitor_id=monitor.id,
        action=MonitorLifecycleAction.ACTIVATE,
        from_status=None,
        to_status=MonitorStatus.ENABLED,
        from_revision=0,
        to_revision=1,
        monitor_version=1,
        actor_id="user-a",
        reason="Confirmed activation.",
        idempotency_key="activation-a",
        occurred_at=NOW,
    )
    state = MonitorScheduleState(
        agent_id="agent-a",
        monitor_id=monitor.id,
        revision=1,
        next_scheduled_at=NOW + timedelta(days=1),
        updated_at=NOW,
    )
    confirmation = MonitorConfirmation(
        id="confirmation-a",
        agent_id="agent-a",
        proposal_id="proposal-a",
        decision=MonitorConfirmationDecision.CONFIRMED,
        candidate_hash=definition.content_hash,
        actor_id="user-a",
        reason="Confirmed activation.",
        decided_at=NOW,
        resulting_monitor_id=monitor.id,
        resulting_version_id=version.id,
    )
    return monitor, version, lifecycle, state, confirmation


def _claim_records() -> tuple[
    MonitorOccurrence,
    MonitorTickLease,
    MonitorTickLeaseGuard,
    MonitorRun,
]:
    key = monitor_occurrence_key(
        agent_id="agent-a",
        monitor_id="monitor-a",
        monitor_version=1,
        kind=MonitorOccurrenceKind.SCHEDULED,
        scheduled_for=NOW,
    )
    occurrence = MonitorOccurrence(
        id=monitor_occurrence_id(key),
        agent_id="agent-a",
        monitor_id="monitor-a",
        monitor_version=1,
        kind=MonitorOccurrenceKind.SCHEDULED,
        scheduled_for=NOW,
        occurrence_key=key,
        trigger_id=monitor_trigger_id(key),
        run_id=monitor_run_id(key),
        created_at=NOW,
    )
    lease = MonitorTickLease(
        id="lease-a",
        agent_id="agent-a",
        monitor_id="monitor-a",
        occurrence_id=occurrence.id,
        holder_id="host-a",
        fencing_token=1,
        claimed_at=NOW,
        expires_at=NOW + timedelta(minutes=1),
    )
    guard = MonitorTickLeaseGuard(
        agent_id="agent-a",
        monitor_id="monitor-a",
        occurrence_id=occurrence.id,
        holder_id="host-a",
        fencing_token=1,
    )
    run = MonitorRun(
        id=occurrence.run_id,
        agent_id="agent-a",
        monitor_id="monitor-a",
        occurrence_id=occurrence.id,
        trigger_id=occurrence.trigger_id,
        attempt=1,
        fencing_token=1,
        status=MonitorRunStatus.RUNNING,
        started_at=NOW,
    )
    return occurrence, lease, guard, run


def test_confirmation_commit_requires_one_exact_activation_identity_and_hash() -> None:
    monitor, version, lifecycle, state, confirmation = _activation()
    commit = MonitorConfirmationCommit(
        proposal_id=confirmation.proposal_id,
        confirmation=confirmation,
        event=_event("event-confirmed", "monitor.confirmed"),
        monitor=monitor,
        version=version,
        lifecycle=lifecycle,
        schedule_state=state,
    )
    assert commit.version is version

    with pytest.raises(ValueError, match="monitor_id does not match monitor"):
        MonitorConfirmationCommit(
            proposal_id=confirmation.proposal_id,
            confirmation=confirmation,
            event=_event(
                "event-wrong-monitor", "monitor.confirmed", monitor_id="other"
            ),
            monitor=monitor,
            version=version,
            lifecycle=lifecycle,
            schedule_state=state,
        )

    other = MonitorDefinition(
        name="Different monitor",
        objective="This is not the confirmed candidate.",
        scope=MonitorScope(source_ids=("source-a",)),
        schedule=CronSchedule("0 9 * * *"),
    )
    mismatched_confirmation = replace(
        confirmation,
        candidate_hash=other.content_hash,
    )
    with pytest.raises(ValueError, match="candidate hash"):
        MonitorConfirmationCommit(
            proposal_id=mismatched_confirmation.proposal_id,
            confirmation=mismatched_confirmation,
            event=_event("event-wrong-hash", "monitor.confirmed"),
            monitor=monitor,
            version=version,
            lifecycle=lifecycle,
            schedule_state=state,
        )


def test_rejected_confirmation_commit_cannot_smuggle_activation_records() -> None:
    monitor, version, lifecycle, state, confirmation = _activation()
    rejected = replace(
        confirmation,
        decision=MonitorConfirmationDecision.REJECTED,
        resulting_monitor_id=None,
        resulting_version_id=None,
    )
    MonitorConfirmationCommit(
        proposal_id=rejected.proposal_id,
        confirmation=rejected,
        event=_event("event-rejected", "monitor.rejected"),
    )
    with pytest.raises(ValueError, match="cannot activate"):
        MonitorConfirmationCommit(
            proposal_id=rejected.proposal_id,
            confirmation=rejected,
            event=_event("event-rejected-smuggle", "monitor.rejected"),
            monitor=monitor,
            version=version,
            lifecycle=lifecycle,
            schedule_state=state,
        )


def test_lifecycle_commit_guards_revision_version_agent_and_monitor_identity() -> None:
    monitor, version, lifecycle, state, _ = _activation()
    MonitorLifecycleCommit(
        monitor=monitor,
        version=version,
        lifecycle=lifecycle,
        schedule_state=state,
        event=_event("event-lifecycle", "monitor.activated"),
    )

    with pytest.raises(ValueError, match="revision"):
        MonitorLifecycleCommit(
            monitor=replace(monitor, revision=2),
            version=version,
            lifecycle=lifecycle,
            schedule_state=state,
            event=_event("event-revision", "monitor.updated"),
        )
    with pytest.raises(ValueError, match="version does not match"):
        MonitorLifecycleCommit(
            monitor=monitor,
            version=replace(version, monitor_id="other"),
            lifecycle=lifecycle,
            schedule_state=state,
            event=_event("event-version", "monitor.updated"),
        )


def test_occurrence_claim_guards_stable_ids_and_fence() -> None:
    occurrence, lease, _, run = _claim_records()
    MonitorOccurrenceClaim(
        occurrence=occurrence,
        lease=lease,
        run=run,
        event=_event("event-claim", "monitor.claimed"),
    )
    with pytest.raises(ValueError, match="fence"):
        MonitorOccurrenceClaim(
            occurrence=occurrence,
            lease=replace(lease, fencing_token=2),
            run=run,
            event=_event("event-stale", "monitor.claimed"),
        )
    with pytest.raises(ValueError, match="identities do not match"):
        MonitorOccurrenceClaim(
            occurrence=occurrence,
            lease=replace(lease, monitor_id="other"),
            run=run,
            event=_event("event-other", "monitor.claimed"),
        )


def test_outcome_commit_guards_fence_run_links_events_and_evidence_finding() -> None:
    occurrence, lease, guard, running = _claim_records()
    completed_at = NOW + timedelta(seconds=10)
    released = replace(
        lease,
        released_at=completed_at,
        release_reason="completed",
    )
    run = replace(
        running,
        status=MonitorRunStatus.SUCCEEDED,
        operation_id="operation-a",
        completed_at=completed_at,
    )
    state = MonitorScheduleState(
        agent_id="agent-a",
        monitor_id="monitor-a",
        revision=2,
        next_scheduled_at=NOW + timedelta(days=1),
        updated_at=completed_at,
        last_scheduled_at=NOW,
        checkpoint_version=1,
        last_occurrence_id=occurrence.id,
        last_run_id=run.id,
        last_operation_id="operation-a",
    )
    checkpoint = MonitorCheckpoint(
        id="checkpoint-a",
        agent_id="agent-a",
        monitor_id="monitor-a",
        version=1,
        run_id=run.id,
        cursor={},
        cursor_hash="sha256:44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        created_at=completed_at,
    )
    finding = MonitorFinding(
        id="finding-a",
        agent_id="agent-a",
        monitor_id="monitor-a",
        occurrence_id=occurrence.id,
        run_id=run.id,
        operation_id="operation-a",
        evidence_id="evidence-a",
        severity=MonitorFindingSeverity.WARNING,
        summary="Threshold matched.",
        details={},
        dedupe_key="dedupe-a",
        created_at=completed_at,
    )
    commit = MonitorOutcomeCommit(
        guard=guard,
        run=run,
        released_lease=released,
        schedule_state=state,
        events=(_event("event-outcome", "monitor.completed"),),
        checkpoint=checkpoint,
        finding=finding,
    )
    assert commit.finding is not None
    assert commit.finding.evidence_id == "evidence-a"

    with pytest.raises(ValueError, match="at least one event"):
        MonitorOutcomeCommit(
            guard=guard,
            run=run,
            released_lease=released,
            schedule_state=state,
            events=(),
        )
    with pytest.raises(ValueError, match="released lease"):
        MonitorOutcomeCommit(
            guard=guard,
            run=run,
            released_lease=replace(released, fencing_token=2),
            schedule_state=state,
            events=(_event("event-stale-outcome", "monitor.completed"),),
        )
    with pytest.raises(ValueError, match="checkpoint does not match run"):
        MonitorOutcomeCommit(
            guard=guard,
            run=run,
            released_lease=released,
            schedule_state=state,
            events=(_event("event-checkpoint", "monitor.completed"),),
            checkpoint=replace(checkpoint, run_id="other-run"),
        )
