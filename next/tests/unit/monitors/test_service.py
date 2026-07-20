from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from daita.monitors import (
    IntervalSchedule,
    MonitorClaimResult,
    MonitorBudgetOverrides,
    MonitorConfirmation,
    MonitorConfirmationCommit,
    MonitorConditionKind,
    MonitorDefinition,
    MonitorInspection,
    MonitorLifecycleCommit,
    MonitorLifecycleError,
    MonitorNotFoundError,
    MonitorOccurrence,
    MonitorOccurrenceClaim,
    MonitorProposal,
    MonitorProposalConflictError,
    MonitorRun,
    MonitorRunStatus,
    MonitorScope,
    MonitorService,
    MonitorStatus,
    MonitorTickLease,
)
from daita.events.models import RuntimeEvent

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)


class MutableClock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def __call__(self) -> datetime:
        return self.value


class StableIds:
    def __init__(self) -> None:
        self.count = 0

    def __call__(self, prefix: str) -> str:
        self.count += 1
        return f"{prefix}-{self.count}"


class FakeMonitorStore:
    def __init__(self) -> None:
        self.proposals: dict[str, MonitorProposal] = {}
        self.proposal_by_key: dict[tuple[str, str], MonitorProposal] = {}
        self.proposal_events: list[RuntimeEvent] = []
        self.confirmations: dict[str, MonitorConfirmation] = {}
        self.confirmation_commits: list[MonitorConfirmationCommit] = []
        self.lifecycle_commits: list[tuple[MonitorLifecycleCommit, int]] = []
        self.lifecycle_by_key: dict[str, MonitorInspection] = {}
        self.inspections: dict[str, MonitorInspection] = {}
        self.claim_calls: list[dict[str, Any]] = []
        self.claim_by_manual_key: dict[str, MonitorClaimResult] = {}

    async def create_monitor_proposal(self, proposal, event):
        self.proposal_events.append(event)
        key = (proposal.agent_id, proposal.idempotency_key)
        existing = self.proposal_by_key.get(key)
        if existing is not None:
            return existing
        self.proposals[proposal.id] = proposal
        self.proposal_by_key[key] = proposal
        return proposal

    async def load_monitor_proposal(self, agent_id, proposal_id):
        proposal = self.proposals.get(proposal_id)
        return (
            proposal if proposal is not None and proposal.agent_id == agent_id else None
        )

    async def load_monitor_confirmation(self, agent_id, proposal_id):
        confirmation = self.confirmations.get(proposal_id)
        if confirmation is not None and confirmation.agent_id == agent_id:
            return confirmation
        return None

    async def list_monitor_proposals(self, agent_id, *, limit):
        return tuple(
            proposal
            for proposal in self.proposals.values()
            if proposal.agent_id == agent_id
        )[:limit]

    async def commit_monitor_confirmation(self, commit):
        self.confirmation_commits.append(commit)
        existing = self.confirmations.get(commit.proposal_id)
        if existing is not None:
            if (
                existing.decision is not commit.confirmation.decision
                or existing.candidate_hash != commit.confirmation.candidate_hash
            ):
                raise AssertionError("divergent confirmation replay")
            if existing.resulting_monitor_id is None:
                return None
            return self.inspections[existing.resulting_monitor_id]
        self.confirmations[commit.proposal_id] = commit.confirmation
        if commit.monitor is None:
            return None
        assert commit.version is not None
        assert commit.lifecycle is not None
        assert commit.schedule_state is not None
        inspection = MonitorInspection(
            monitor=commit.monitor,
            versions=(commit.version,),
            lifecycle=(commit.lifecycle,),
            schedule_state=commit.schedule_state,
            proposals=(self.proposals[commit.proposal_id],),
            confirmations=(commit.confirmation,),
        )
        self.inspections[commit.monitor.id] = inspection
        return inspection

    async def inspect_monitor(self, agent_id, monitor_id):
        inspection = self.inspections.get(monitor_id)
        if inspection is not None and inspection.monitor.agent_id == agent_id:
            return inspection
        return None

    async def list_monitors(self, agent_id, *, statuses, limit):
        return tuple(
            inspection.monitor
            for inspection in self.inspections.values()
            if inspection.monitor.agent_id == agent_id
            and inspection.monitor.status in statuses
        )[:limit]

    async def commit_monitor_lifecycle(self, commit, *, expected_revision):
        existing = self.lifecycle_by_key.get(commit.lifecycle.idempotency_key)
        if existing is not None:
            return existing
        current = self.inspections[commit.monitor.id]
        assert current.monitor.revision == expected_revision
        self.lifecycle_commits.append((commit, expected_revision))
        updated = replace(
            current,
            monitor=commit.monitor,
            lifecycle=(*current.lifecycle, commit.lifecycle),
            schedule_state=commit.schedule_state,
        )
        self.inspections[commit.monitor.id] = updated
        self.lifecycle_by_key[commit.lifecycle.idempotency_key] = updated
        return updated

    async def list_due_monitors(self, agent_id, *, now, limit):
        return ()

    async def claim_monitor_occurrence(
        self,
        claim: MonitorOccurrenceClaim,
        *,
        expected_monitor_revision,
        expected_schedule_revision,
        checked_at,
    ):
        self.claim_calls.append(
            {
                "claim": claim,
                "expected_monitor_revision": expected_monitor_revision,
                "expected_schedule_revision": expected_schedule_revision,
                "checked_at": checked_at,
            }
        )
        manual_key = claim.occurrence.manual_key
        assert manual_key is not None
        existing = self.claim_by_manual_key.get(manual_key)
        if existing is not None:
            assert claim.occurrence == existing.occurrence
            assert claim.lease.fencing_token == existing.lease.fencing_token + 1
            assert claim.run.attempt == existing.run.attempt + 1
        result = MonitorClaimResult(claim.occurrence, claim.lease, claim.run)
        self.claim_by_manual_key[manual_key] = result
        return result

    async def load_monitor_claim_by_manual_key(
        self,
        agent_id: str,
        monitor_id: str,
        manual_key: str,
    ) -> MonitorClaimResult | None:
        result = self.claim_by_manual_key.get(manual_key)
        if result is None:
            return None
        if (
            result.occurrence.agent_id != agent_id
            or result.occurrence.monitor_id != monitor_id
        ):
            return None
        return result

    async def load_occurrence_by_trigger(self, agent_id, trigger_id):
        return None

    async def commit_monitor_outcome(self, *args, **kwargs):
        raise AssertionError("service must not commit scheduler outcomes")


def _definition() -> MonitorDefinition:
    return MonitorDefinition(
        name="Orders backlog",
        objective="Inspect orders and record a finding when backlog is too high.",
        scope=MonitorScope(
            source_ids=("source-orders",),
            resource_ids=("resource-orders",),
        ),
        schedule=IntervalSchedule(interval_seconds=300, anchor_at=NOW),
    )


def _service(store: FakeMonitorStore, clock: MutableClock) -> MonitorService:
    return MonitorService(
        agent_id="agent-1",
        store=store,
        clock=clock,
        id_factory=StableIds(),
    )


async def _activated(
    store: FakeMonitorStore,
    clock: MutableClock,
) -> tuple[MonitorService, MonitorInspection]:
    service = _service(store, clock)
    proposal = await service.propose(
        "orders-backlog",
        _definition(),
        idempotency_key="request-create-orders",
        source_operation_id="operation-create-orders",
    )
    inspection = await service.confirm(
        proposal.id,
        candidate_hash=proposal.candidate_hash,
        actor_id="user-1",
        reason="Schedule and scope confirmed.",
    )
    return service, inspection


async def test_proposal_is_inert_correlated_and_store_idempotent() -> None:
    store = FakeMonitorStore()
    clock = MutableClock()
    service = _service(store, clock)
    definition = _definition()

    first = await service.propose(
        "orders-backlog",
        definition,
        idempotency_key="request-create-orders",
        source_operation_id="operation-create-orders",
    )
    clock.value += timedelta(seconds=1)
    replay = await service.propose(
        "orders-backlog",
        definition,
        idempotency_key="request-create-orders",
        source_operation_id="operation-create-orders",
    )

    assert replay == first
    assert store.inspections == {}
    assert len(store.proposal_events) == 2
    event = store.proposal_events[0]
    assert event.type == "monitor.proposed"
    assert event.agent_id == "agent-1"
    assert event.monitor_id == "orders-backlog"
    assert event.operation_id is None
    assert event.payload["source_operation_id"] == "operation-create-orders"
    assert event.payload["candidate_hash"] == definition.content_hash


async def test_natural_monitor_request_creates_only_safe_typed_definition() -> None:
    store = FakeMonitorStore()
    service = _service(store, MutableClock())

    proposal = await service.propose_natural(
        "orders-threshold",
        "Every 5 minutes, Count open orders for source source-orders "
        "when rows.0.total > 10",
        idempotency_key="natural-orders-1",
    )

    definition = proposal.candidate
    assert definition.objective == "Count open orders"
    assert definition.scope.source_ids == ("source-orders",)
    assert isinstance(definition.schedule, IntervalSchedule)
    assert definition.schedule.interval_seconds == 300
    assert definition.schedule.anchor_at == NOW
    assert definition.condition.kind is MonitorConditionKind.THRESHOLD
    assert definition.condition.expression == "rows.0.total"
    assert dict(definition.condition.configuration) == {
        "operator": "gt",
        "value": 10,
    }
    assert store.inspections == {}


@pytest.mark.parametrize(
    "message",
    (
        "SELECT * FROM orders",
        "Every 5 minutes, run os.system for source source-orders when x eval 1",
        "Every 999999 hours, Count orders for source source-orders",
    ),
)
async def test_natural_monitor_request_rejects_unsupported_or_unbounded_text(
    message: str,
) -> None:
    store = FakeMonitorStore()
    service = _service(store, MutableClock())

    with pytest.raises(ValueError):
        await service.propose_natural(
            "orders-threshold",
            message,
            idempotency_key="natural-orders-invalid",
        )

    assert store.proposals == {}


@pytest.mark.parametrize(
    "definition",
    [
        replace(
            _definition(),
            budget_overrides=MonitorBudgetOverrides(max_turns=9),
        ),
        replace(_definition(), policy_overrides={"allow_destructive": True}),
        replace(_definition(), operation_template={"domain": "custom"}),
    ],
)
async def test_proposal_rejects_settings_that_expand_agent_authority(
    definition: MonitorDefinition,
) -> None:
    store = FakeMonitorStore()
    service = _service(store, MutableClock())

    with pytest.raises(ValueError, match="monitor"):
        await service.propose(
            "orders-backlog",
            definition,
            idempotency_key="request-invalid-monitor",
        )

    assert store.proposals == {}
    assert store.proposal_events == []


async def test_confirmation_requires_exact_candidate_hash_and_activates_once() -> None:
    store = FakeMonitorStore()
    clock = MutableClock()
    service = _service(store, clock)
    proposal = await service.propose(
        "orders-backlog",
        _definition(),
        idempotency_key="request-create-orders",
    )

    with pytest.raises(MonitorProposalConflictError, match="candidate_hash"):
        await service.confirm(
            proposal.id,
            candidate_hash="sha256:" + "0" * 64,
            actor_id="user-1",
            reason="Wrong preview.",
        )
    assert store.confirmation_commits == []

    inspection = await service.confirm(
        proposal.id,
        candidate_hash=proposal.candidate_hash,
        actor_id="user-1",
        reason="Confirmed exact preview.",
    )
    commit = store.confirmation_commits[-1]

    assert inspection.monitor.status is MonitorStatus.ENABLED
    assert inspection.monitor.current_version == 1
    assert inspection.monitor.revision == 1
    assert inspection.versions[0].content_hash == proposal.candidate_hash
    assert inspection.schedule_state.next_scheduled_at == NOW + timedelta(minutes=5)
    assert commit.event.type == "monitor.activated"
    assert commit.event.monitor_id == inspection.monitor.id
    assert commit.lifecycle is not None
    assert commit.lifecycle.from_revision == 0
    assert commit.lifecycle.to_revision == 1


async def test_rejection_keeps_proposal_inert() -> None:
    store = FakeMonitorStore()
    clock = MutableClock()
    service = _service(store, clock)
    proposal = await service.propose(
        "orders-backlog",
        _definition(),
        idempotency_key="request-reject-orders",
    )

    confirmation = await service.reject(
        proposal.id,
        candidate_hash=proposal.candidate_hash,
        actor_id="user-1",
        reason="Do not activate this schedule.",
    )

    assert confirmation.resulting_monitor_id is None
    assert store.inspections == {}
    assert store.confirmation_commits[-1].event.type == "monitor.rejected"
    assert store.confirmation_commits[-1].event.monitor_id == "orders-backlog"


async def test_pause_resume_delete_are_soft_versioned_schedule_transitions() -> None:
    store = FakeMonitorStore()
    clock = MutableClock()
    service, activated = await _activated(store, clock)
    store.inspections[activated.monitor.id] = replace(
        activated,
        schedule_state=replace(
            activated.schedule_state,
            cooldown_until=NOW + timedelta(hours=1),
            backoff_until=NOW + timedelta(minutes=30),
            consecutive_failures=2,
            consecutive_matches=3,
        ),
    )

    clock.value = NOW + timedelta(minutes=1)
    paused = await service.pause(
        activated.monitor.id,
        actor_id="user-1",
        reason="Maintenance window.",
        idempotency_key="pause-orders",
    )
    assert paused.monitor.status is MonitorStatus.PAUSED
    assert paused.monitor.paused_at == clock.value
    assert paused.schedule_state.next_scheduled_at == NOW + timedelta(minutes=5)
    paused_replay = await service.pause(
        activated.monitor.id,
        actor_id="user-1",
        reason="Maintenance window.",
        idempotency_key="pause-orders",
    )
    assert paused_replay == paused

    clock.value = NOW + timedelta(minutes=20)
    resumed = await service.resume(
        activated.monitor.id,
        actor_id="user-1",
        reason="Maintenance complete.",
        idempotency_key="resume-orders",
    )
    assert resumed.monitor.status is MonitorStatus.ENABLED
    assert resumed.monitor.paused_at is None
    assert resumed.schedule_state.next_scheduled_at == NOW + timedelta(minutes=25)
    assert resumed.schedule_state.cooldown_until == NOW + timedelta(hours=1)
    assert resumed.schedule_state.backoff_until == NOW + timedelta(minutes=30)
    assert resumed.schedule_state.consecutive_failures == 2
    assert resumed.schedule_state.consecutive_matches == 3

    clock.value += timedelta(minutes=1)
    deleted = await service.delete(
        activated.monitor.id,
        actor_id="user-1",
        reason="Monitor retired.",
        idempotency_key="delete-orders",
    )
    assert deleted.monitor.status is MonitorStatus.DELETED
    assert deleted.monitor.deleted_at == clock.value
    assert deleted.schedule_state.next_scheduled_at is None
    assert deleted.versions == activated.versions
    assert len(deleted.lifecycle) == 4
    assert [commit.event.type for commit, _ in store.lifecycle_commits] == [
        "monitor.pause",
        "monitor.resume",
        "monitor.delete",
    ]
    assert await service.list() == ()
    assert await service.list(include_deleted=True) == (deleted.monitor,)


async def test_run_now_claims_manual_occurrence_without_executing_or_changing_cadence() -> (
    None
):
    store = FakeMonitorStore()
    clock = MutableClock()
    service, activated = await _activated(store, clock)
    clock.value += timedelta(minutes=1)
    paused = await service.pause(
        activated.monitor.id,
        actor_id="user-1",
        reason="Pause automatic cadence.",
        idempotency_key="pause-before-manual",
    )
    lifecycle_count = len(store.lifecycle_commits)

    first = await service.run_now(
        paused.monitor.id,
        idempotency_key="manual-run-1",
        holder_id="host-1",
        lease_seconds=30,
    )
    clock.value += timedelta(seconds=5)
    replay = await service.run_now(
        paused.monitor.id,
        idempotency_key="manual-run-1",
        holder_id="host-1",
        lease_seconds=30,
    )

    assert replay == first
    assert len(store.claim_calls) == 1
    assert len(store.lifecycle_commits) == lifecycle_count
    assert store.inspections[paused.monitor.id].schedule_state == paused.schedule_state
    claim = store.claim_calls[0]["claim"]
    assert isinstance(claim.occurrence, MonitorOccurrence)
    assert isinstance(claim.lease, MonitorTickLease)
    assert isinstance(claim.run, MonitorRun)
    assert claim.occurrence.manual_key == "manual-run-1"
    assert claim.event.type == "monitor.claimed"
    assert claim.event.monitor_id == paused.monitor.id
    assert claim.event.operation_id is None
    assert claim.run.operation_id is None


@pytest.mark.parametrize(
    ("prior_status", "operation_id"),
    [
        (MonitorRunStatus.PENDING, None),
        (MonitorRunStatus.WAITING, "operation-waiting-1"),
    ],
)
async def test_run_now_reclaims_expired_same_key_with_next_fence_and_attempt(
    prior_status: MonitorRunStatus,
    operation_id: str | None,
) -> None:
    store = FakeMonitorStore()
    clock = MutableClock()
    service, activated = await _activated(store, clock)
    first = await service.run_now(
        activated.monitor.id,
        idempotency_key="manual-recover-1",
        holder_id="host-before-crash",
        lease_seconds=30,
    )
    store.claim_by_manual_key["manual-recover-1"] = MonitorClaimResult(
        occurrence=first.occurrence,
        lease=first.lease,
        run=replace(
            first.run,
            status=prior_status,
            operation_id=operation_id,
        ),
    )
    clock.value += timedelta(seconds=31)

    recovered = await service.run_now(
        activated.monitor.id,
        idempotency_key="manual-recover-1",
        holder_id="host-after-crash",
        lease_seconds=30,
    )

    assert recovered.occurrence == first.occurrence
    assert recovered.lease.holder_id == "host-after-crash"
    assert recovered.lease.fencing_token == 2
    assert recovered.run.attempt == 2
    assert recovered.run.fencing_token == 2
    assert recovered.run.status is MonitorRunStatus.PENDING
    assert recovered.run.operation_id == operation_id
    assert len(store.claim_calls) == 2
    reclaim = store.claim_calls[-1]["claim"]
    assert reclaim.event.payload["fencing_token"] == 2
    assert reclaim.event.payload["operation_id"] == operation_id


async def test_missing_and_deleted_monitors_cannot_run() -> None:
    store = FakeMonitorStore()
    clock = MutableClock()
    service = _service(store, clock)
    with pytest.raises(MonitorNotFoundError):
        await service.inspect("missing-monitor")

    service, activated = await _activated(store, clock)
    deleted = await service.delete(
        activated.monitor.id,
        actor_id="user-1",
        reason="Retired.",
        idempotency_key="delete-before-run",
    )
    with pytest.raises(MonitorLifecycleError, match="deleted"):
        await service.run_now(
            deleted.monitor.id,
            idempotency_key="manual-after-delete",
        )
