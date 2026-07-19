"""Portable atomic persistence contract for durable monitor lifecycles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from ..events.models import RuntimeEvent
from .models import (
    Monitor,
    MonitorCheckpoint,
    MonitorConfirmation,
    MonitorConfirmationDecision,
    MonitorFinding,
    MonitorInspection,
    MonitorLifecycleRecord,
    MonitorOccurrence,
    MonitorProposal,
    MonitorRun,
    MonitorRunStatus,
    MonitorScheduleState,
    MonitorStatus,
    MonitorTickLease,
    MonitorTickLeaseGuard,
    MonitorVersion,
)


class MonitorStoreError(RuntimeError):
    """Base class for portable monitor-store failures."""


class MonitorNotFoundError(MonitorStoreError):
    def __init__(self, agent_id: str, monitor_id: str) -> None:
        self.agent_id = agent_id
        self.monitor_id = monitor_id
        super().__init__(f"unknown monitor for {agent_id}: {monitor_id}")


class MonitorProposalNotFoundError(MonitorStoreError):
    def __init__(self, agent_id: str, proposal_id: str) -> None:
        self.agent_id = agent_id
        self.proposal_id = proposal_id
        super().__init__(f"unknown monitor proposal for {agent_id}: {proposal_id}")


class MonitorConflictError(MonitorStoreError):
    def __init__(
        self,
        monitor_id: str,
        *,
        expected_revision: int,
        actual_revision: int,
    ) -> None:
        self.monitor_id = monitor_id
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"monitor {monitor_id} revision conflict: expected "
            f"{expected_revision}, found {actual_revision}"
        )


class MonitorProposalConflictError(MonitorStoreError):
    def __init__(self, proposal_id: str, reason: str) -> None:
        self.proposal_id = proposal_id
        self.reason = reason
        super().__init__(f"monitor proposal {proposal_id} conflict: {reason}")


class MonitorTickClaimConflictError(MonitorStoreError):
    def __init__(
        self,
        occurrence_id: str,
        *,
        holder_id: str,
        fencing_token: int,
        expires_at: datetime,
    ) -> None:
        self.occurrence_id = occurrence_id
        self.holder_id = holder_id
        self.fencing_token = fencing_token
        self.expires_at = expires_at
        super().__init__(
            f"monitor occurrence {occurrence_id} is claimed by {holder_id} "
            f"with fence {fencing_token} until {expires_at.isoformat()}"
        )


class StaleMonitorFenceError(MonitorStoreError):
    def __init__(
        self,
        occurrence_id: str,
        *,
        expected_fencing_token: int,
        actual_fencing_token: int,
    ) -> None:
        self.occurrence_id = occurrence_id
        self.expected_fencing_token = expected_fencing_token
        self.actual_fencing_token = actual_fencing_token
        super().__init__(
            f"monitor occurrence {occurrence_id} fence is stale: expected "
            f"{expected_fencing_token}, found {actual_fencing_token}"
        )


class ExpiredMonitorLeaseError(MonitorStoreError):
    def __init__(
        self,
        occurrence_id: str,
        *,
        fencing_token: int,
        expires_at: datetime,
        checked_at: datetime,
    ) -> None:
        self.occurrence_id = occurrence_id
        self.fencing_token = fencing_token
        self.expires_at = expires_at
        self.checked_at = checked_at
        super().__init__(
            f"monitor occurrence {occurrence_id} fence {fencing_token} expired at "
            f"{expires_at.isoformat()} before {checked_at.isoformat()}"
        )


def _monitor_event(event: RuntimeEvent, monitor_id: str, event_name: str) -> None:
    if not isinstance(event, RuntimeEvent):
        raise TypeError(f"{event_name} must be a RuntimeEvent")
    if event.monitor_id != monitor_id:
        raise ValueError(f"{event_name} monitor_id does not match monitor")


@dataclass(frozen=True, slots=True)
class MonitorConfirmationCommit:
    proposal_id: str
    confirmation: MonitorConfirmation
    event: RuntimeEvent
    monitor: Monitor | None = None
    version: MonitorVersion | None = None
    lifecycle: MonitorLifecycleRecord | None = None
    schedule_state: MonitorScheduleState | None = None

    def __post_init__(self) -> None:
        if self.confirmation.proposal_id != self.proposal_id:
            raise ValueError("confirmation proposal_id does not match commit")
        confirmed = self.confirmation.decision is MonitorConfirmationDecision.CONFIRMED
        activation = (
            self.monitor,
            self.version,
            self.lifecycle,
            self.schedule_state,
        )
        if confirmed and any(value is None for value in activation):
            raise ValueError("confirmed monitor commit requires activation records")
        if not confirmed and any(value is not None for value in activation):
            raise ValueError("rejected monitor commit cannot activate records")
        monitor_id = (
            self.confirmation.resulting_monitor_id
            if confirmed
            else self.event.monitor_id
        )
        if monitor_id is not None:
            _monitor_event(self.event, monitor_id, "confirmation event")
        if confirmed:
            assert self.monitor is not None
            assert self.version is not None
            assert self.lifecycle is not None
            assert self.schedule_state is not None
            identities = {
                self.monitor.agent_id,
                self.version.agent_id,
                self.lifecycle.agent_id,
                self.schedule_state.agent_id,
                self.confirmation.agent_id,
                self.event.agent_id,
            }
            monitor_ids = {
                self.monitor.id,
                self.version.monitor_id,
                self.lifecycle.monitor_id,
                self.schedule_state.monitor_id,
                self.confirmation.resulting_monitor_id,
                self.event.monitor_id,
            }
            if len(identities) != 1 or len(monitor_ids) != 1:
                raise ValueError("confirmation activation identities do not match")
            if self.version.version != 1 or self.monitor.current_version != 1:
                raise ValueError("monitor activation must create version one")
            if self.confirmation.candidate_hash != self.version.content_hash:
                raise ValueError(
                    "confirmation candidate hash does not match activated version"
                )
            if self.lifecycle.from_revision != 0 or self.lifecycle.to_revision != 1:
                raise ValueError(
                    "monitor activation must create lifecycle revision one"
                )


@dataclass(frozen=True, slots=True)
class MonitorLifecycleCommit:
    monitor: Monitor
    lifecycle: MonitorLifecycleRecord
    schedule_state: MonitorScheduleState
    event: RuntimeEvent
    version: MonitorVersion | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.monitor, Monitor):
            raise TypeError("lifecycle commit monitor must be Monitor")
        if not isinstance(self.lifecycle, MonitorLifecycleRecord):
            raise TypeError("lifecycle commit lifecycle must be MonitorLifecycleRecord")
        if not isinstance(self.schedule_state, MonitorScheduleState):
            raise TypeError(
                "lifecycle commit schedule_state must be MonitorScheduleState"
            )
        agent_ids = {
            self.monitor.agent_id,
            self.lifecycle.agent_id,
            self.schedule_state.agent_id,
            self.event.agent_id,
        }
        monitor_ids = {
            self.monitor.id,
            self.lifecycle.monitor_id,
            self.schedule_state.monitor_id,
            self.event.monitor_id,
        }
        if len(agent_ids) != 1 or len(monitor_ids) != 1:
            raise ValueError("lifecycle commit identities do not match")
        _monitor_event(self.event, self.monitor.id, "lifecycle event")
        if self.monitor.revision != self.lifecycle.to_revision:
            raise ValueError(
                "lifecycle commit monitor revision does not match transition"
            )
        if self.version is not None:
            if (
                self.version.agent_id != self.monitor.agent_id
                or self.version.monitor_id != self.monitor.id
                or self.version.version != self.monitor.current_version
            ):
                raise ValueError("lifecycle commit version does not match monitor")


@dataclass(frozen=True, slots=True)
class MonitorOccurrenceClaim:
    occurrence: MonitorOccurrence
    lease: MonitorTickLease
    run: MonitorRun
    event: RuntimeEvent

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence, MonitorOccurrence):
            raise TypeError("claim occurrence must be MonitorOccurrence")
        if not isinstance(self.lease, MonitorTickLease):
            raise TypeError("claim lease must be MonitorTickLease")
        if not isinstance(self.run, MonitorRun):
            raise TypeError("claim run must be MonitorRun")
        agent_ids = {
            self.occurrence.agent_id,
            self.lease.agent_id,
            self.run.agent_id,
            self.event.agent_id,
        }
        monitor_ids = {
            self.occurrence.monitor_id,
            self.lease.monitor_id,
            self.run.monitor_id,
            self.event.monitor_id,
        }
        occurrence_ids = {
            self.occurrence.id,
            self.lease.occurrence_id,
            self.run.occurrence_id,
        }
        if len(agent_ids) != 1 or len(monitor_ids) != 1 or len(occurrence_ids) != 1:
            raise ValueError("monitor claim identities do not match")
        if self.run.id != self.occurrence.run_id:
            raise ValueError("monitor claim run does not use stable occurrence run_id")
        if self.run.trigger_id != self.occurrence.trigger_id:
            raise ValueError("monitor claim run does not use stable trigger_id")
        if self.run.fencing_token != self.lease.fencing_token:
            raise ValueError("monitor claim run and lease fence do not match")
        _monitor_event(self.event, self.occurrence.monitor_id, "claim event")


@dataclass(frozen=True, slots=True)
class MonitorClaimResult:
    occurrence: MonitorOccurrence
    lease: MonitorTickLease
    run: MonitorRun

    def __post_init__(self) -> None:
        if not isinstance(self.occurrence, MonitorOccurrence):
            raise TypeError("claim result occurrence must be MonitorOccurrence")
        if not isinstance(self.lease, MonitorTickLease):
            raise TypeError("claim result lease must be MonitorTickLease")
        if not isinstance(self.run, MonitorRun):
            raise TypeError("claim result run must be MonitorRun")
        identities = {
            (
                self.occurrence.agent_id,
                self.occurrence.monitor_id,
                self.occurrence.id,
            ),
            (self.lease.agent_id, self.lease.monitor_id, self.lease.occurrence_id),
            (self.run.agent_id, self.run.monitor_id, self.run.occurrence_id),
        }
        if len(identities) != 1:
            raise ValueError("monitor claim result identities do not match")
        if (
            self.run.id != self.occurrence.run_id
            or self.run.trigger_id != self.occurrence.trigger_id
            or self.run.fencing_token != self.lease.fencing_token
        ):
            raise ValueError(
                "monitor claim result run and lease do not match occurrence"
            )


@dataclass(frozen=True, slots=True)
class MonitorOutcomeCommit:
    guard: MonitorTickLeaseGuard
    run: MonitorRun
    released_lease: MonitorTickLease
    schedule_state: MonitorScheduleState
    events: tuple[RuntimeEvent, ...]
    checkpoint: MonitorCheckpoint | None = None
    finding: MonitorFinding | None = None

    def __post_init__(self) -> None:
        if self.released_lease.released_at is None:
            raise ValueError("monitor outcome requires a released lease")
        identities = {
            (self.guard.agent_id, self.guard.monitor_id, self.guard.occurrence_id),
            (self.run.agent_id, self.run.monitor_id, self.run.occurrence_id),
            (
                self.released_lease.agent_id,
                self.released_lease.monitor_id,
                self.released_lease.occurrence_id,
            ),
            (
                self.schedule_state.agent_id,
                self.schedule_state.monitor_id,
                self.guard.occurrence_id,
            ),
        }
        if len(identities) != 1:
            raise ValueError("monitor outcome identities do not match")
        if (
            self.released_lease.holder_id != self.guard.holder_id
            or self.released_lease.fencing_token != self.guard.fencing_token
            or self.run.fencing_token != self.guard.fencing_token
        ):
            raise ValueError("released lease does not match outcome guard")
        if self.run.status in {MonitorRunStatus.PENDING, MonitorRunStatus.RUNNING}:
            raise ValueError("monitor outcome run must be waiting or terminal")
        events = tuple(self.events)
        if not events:
            raise ValueError("monitor outcome requires at least one event")
        for event in events:
            _monitor_event(event, self.guard.monitor_id, "outcome event")
        object.__setattr__(self, "events", events)
        if self.checkpoint is not None and (
            self.checkpoint.agent_id != self.run.agent_id
            or self.checkpoint.monitor_id != self.run.monitor_id
            or self.checkpoint.run_id != self.run.id
        ):
            raise ValueError("outcome checkpoint does not match run")
        if self.finding is not None and (
            self.finding.agent_id != self.run.agent_id
            or self.finding.monitor_id != self.run.monitor_id
            or self.finding.occurrence_id != self.run.occurrence_id
            or self.finding.run_id != self.run.id
            or self.run.operation_id is None
            or self.finding.operation_id != self.run.operation_id
        ):
            raise ValueError("outcome finding does not match run")


@dataclass(frozen=True, slots=True)
class MonitorOutcomeResult:
    inspection: MonitorInspection
    run: MonitorRun
    finding: MonitorFinding | None


class MonitorStore(Protocol):
    async def create_monitor_proposal(
        self,
        proposal: MonitorProposal,
        event: RuntimeEvent,
    ) -> MonitorProposal: ...

    async def load_monitor_proposal(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> MonitorProposal | None: ...

    async def load_monitor_confirmation(
        self,
        agent_id: str,
        proposal_id: str,
    ) -> MonitorConfirmation | None: ...

    async def list_monitor_proposals(
        self,
        agent_id: str,
        *,
        limit: int,
    ) -> tuple[MonitorProposal, ...]: ...

    async def commit_monitor_confirmation(
        self,
        commit: MonitorConfirmationCommit,
    ) -> MonitorInspection | None: ...

    async def inspect_monitor(
        self,
        agent_id: str,
        monitor_id: str,
    ) -> MonitorInspection | None: ...

    async def list_monitors(
        self,
        agent_id: str,
        *,
        statuses: tuple[MonitorStatus, ...],
        limit: int,
    ) -> tuple[Monitor, ...]: ...

    async def commit_monitor_lifecycle(
        self,
        commit: MonitorLifecycleCommit,
        *,
        expected_revision: int,
    ) -> MonitorInspection: ...

    async def list_due_monitors(
        self,
        agent_id: str,
        *,
        now: datetime,
        limit: int,
    ) -> tuple[MonitorInspection, ...]: ...

    async def claim_monitor_occurrence(
        self,
        claim: MonitorOccurrenceClaim,
        *,
        expected_monitor_revision: int,
        expected_schedule_revision: int,
        checked_at: datetime,
    ) -> MonitorClaimResult: ...

    async def load_monitor_claim_by_manual_key(
        self,
        agent_id: str,
        monitor_id: str,
        manual_key: str,
    ) -> MonitorClaimResult | None: ...

    async def load_occurrence_by_trigger(
        self,
        agent_id: str,
        trigger_id: str,
    ) -> MonitorOccurrence | None: ...

    async def commit_monitor_outcome(
        self,
        commit: MonitorOutcomeCommit,
        *,
        expected_monitor_revision: int,
        expected_schedule_revision: int,
        checked_at: datetime,
    ) -> MonitorOutcomeResult: ...
