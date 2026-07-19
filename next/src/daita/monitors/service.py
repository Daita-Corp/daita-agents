"""Monitor lifecycle coordination without scheduling or execution ownership."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import math
import re
from uuid import uuid4

from ..events.models import RuntimeEvent
from .models import (
    Monitor,
    MonitorConfirmation,
    MonitorConfirmationDecision,
    MonitorDefinition,
    MonitorInspection,
    MonitorLifecycleAction,
    MonitorLifecycleRecord,
    MonitorOccurrence,
    MonitorOccurrenceKind,
    MonitorProposal,
    MonitorRun,
    MonitorRunStatus,
    MonitorScheduleState,
    MonitorStatus,
    MonitorTickLease,
    MonitorVersion,
    monitor_occurrence_id,
    monitor_occurrence_key,
    monitor_run_id,
    monitor_trigger_id,
)
from .store import (
    MonitorClaimResult,
    MonitorConfirmationCommit,
    MonitorLifecycleCommit,
    MonitorNotFoundError,
    MonitorOccurrenceClaim,
    MonitorProposalConflictError,
    MonitorProposalNotFoundError,
    MonitorStore,
)

_IDENTITY = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}\Z")
_MAX_LIST_LIMIT = 1_000
_MAX_LEASE_SECONDS = 300.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _identity(value: str, field_name: str) -> str:
    if not isinstance(value, str) or _IDENTITY.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a bounded stable identifier")
    return value


def _optional_identity(value: str | None, field_name: str) -> str | None:
    if value is not None:
        return _identity(value, field_name)
    return None


def _list_limit(value: int) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 1
        or value > _MAX_LIST_LIMIT
    ):
        raise ValueError(f"limit must be an integer from 1 through {_MAX_LIST_LIMIT}")
    return value


class MonitorServiceContractError(RuntimeError):
    """Raised when a monitor store returns records outside its contract."""


class MonitorLifecycleError(RuntimeError):
    """Raised when a requested lifecycle transition is not meaningful."""


class MonitorService:
    """Construct durable monitor lifecycle records around a narrow store.

    The service validates and correlates lifecycle intent.  The store remains
    responsible for idempotency, optimistic concurrency, and atomic state/event
    commits.  This owner never starts a scheduler or executes monitor work.
    """

    def __init__(
        self,
        *,
        agent_id: str,
        store: MonitorStore,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
    ) -> None:
        self._agent_id = _identity(agent_id, "monitor-service agent_id")
        for method_name in (
            "create_monitor_proposal",
            "load_monitor_proposal",
            "load_monitor_confirmation",
            "list_monitor_proposals",
            "commit_monitor_confirmation",
            "inspect_monitor",
            "list_monitors",
            "commit_monitor_lifecycle",
            "claim_monitor_occurrence",
            "load_monitor_claim_by_manual_key",
        ):
            if not callable(getattr(store, method_name, None)):
                raise TypeError(f"monitor store must provide {method_name}")
        if not callable(clock):
            raise TypeError("monitor clock must be callable")
        if not callable(id_factory):
            raise TypeError("monitor id_factory must be callable")
        self._store = store
        self._clock = clock
        self._id_factory = id_factory

    async def propose(
        self,
        monitor_id: str,
        definition: MonitorDefinition,
        *,
        idempotency_key: str,
        source_operation_id: str | None = None,
    ) -> MonitorProposal:
        """Persist an inert candidate without activating a schedule."""

        monitor_id = _identity(monitor_id, "proposal monitor_id")
        idempotency_key = _identity(
            idempotency_key,
            "proposal idempotency_key",
        )
        source_operation_id = _optional_identity(
            source_operation_id,
            "proposal source_operation_id",
        )
        if not isinstance(definition, MonitorDefinition):
            raise TypeError("definition must be a MonitorDefinition")
        now = self._now()
        proposal = MonitorProposal(
            id=self._id("monitor-proposal"),
            agent_id=self._agent_id,
            intended_monitor_id=monitor_id,
            idempotency_key=idempotency_key,
            candidate=definition,
            candidate_hash=definition.content_hash,
            created_at=now,
            source_operation_id=source_operation_id,
        )
        stored = await self._store.create_monitor_proposal(
            proposal,
            self._event(
                "monitor.proposed",
                now=now,
                monitor_id=monitor_id,
                operation_id=source_operation_id,
                payload={
                    "candidate_hash": proposal.candidate_hash,
                    "idempotency_key": idempotency_key,
                    "proposal_id": proposal.id,
                },
            ),
        )
        self._validate_proposal_replay(proposal, stored)
        return stored

    async def confirm(
        self,
        proposal_id: str,
        *,
        candidate_hash: str,
        actor_id: str,
        reason: str,
    ) -> MonitorInspection:
        """Confirm the exact inert candidate and atomically activate version one."""

        proposal = await self._load_proposal(proposal_id)
        if candidate_hash != proposal.candidate_hash:
            raise MonitorProposalConflictError(
                proposal.id,
                "candidate_hash does not match the persisted proposal",
            )
        actor_id = _identity(actor_id, "confirmation actor_id")
        now = self._now()
        version_id = self._id("monitor-version")
        confirmation = MonitorConfirmation(
            id=self._id("monitor-confirmation"),
            agent_id=self._agent_id,
            proposal_id=proposal.id,
            decision=MonitorConfirmationDecision.CONFIRMED,
            candidate_hash=proposal.candidate_hash,
            actor_id=actor_id,
            reason=reason,
            decided_at=now,
            resulting_monitor_id=proposal.intended_monitor_id,
            resulting_version_id=version_id,
        )
        monitor = Monitor(
            id=proposal.intended_monitor_id,
            agent_id=self._agent_id,
            status=MonitorStatus.ENABLED,
            current_version=1,
            revision=1,
            created_at=now,
            updated_at=now,
        )
        version = MonitorVersion(
            id=version_id,
            agent_id=self._agent_id,
            monitor_id=monitor.id,
            version=1,
            definition=proposal.candidate,
            content_hash=proposal.candidate_hash,
            proposal_id=proposal.id,
            created_at=now,
            source_operation_id=proposal.source_operation_id,
        )
        lifecycle = MonitorLifecycleRecord(
            id=self._id("monitor-lifecycle"),
            agent_id=self._agent_id,
            monitor_id=monitor.id,
            action=MonitorLifecycleAction.ACTIVATE,
            from_status=None,
            to_status=MonitorStatus.ENABLED,
            from_revision=0,
            to_revision=1,
            monitor_version=1,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=proposal.idempotency_key,
            occurred_at=now,
            operation_id=proposal.source_operation_id,
        )
        next_scheduled_at = proposal.candidate.schedule.next_due_at(now)
        schedule_state = MonitorScheduleState(
            agent_id=self._agent_id,
            monitor_id=monitor.id,
            revision=1,
            next_scheduled_at=next_scheduled_at,
            updated_at=now,
        )
        inspection = await self._store.commit_monitor_confirmation(
            MonitorConfirmationCommit(
                proposal_id=proposal.id,
                confirmation=confirmation,
                monitor=monitor,
                version=version,
                lifecycle=lifecycle,
                schedule_state=schedule_state,
                event=self._event(
                    "monitor.activated",
                    now=now,
                    monitor_id=monitor.id,
                    operation_id=proposal.source_operation_id,
                    payload={
                        "candidate_hash": proposal.candidate_hash,
                        "monitor_version": 1,
                        "next_scheduled_at": next_scheduled_at.isoformat(),
                        "proposal_id": proposal.id,
                    },
                ),
            )
        )
        if inspection is None:
            raise MonitorServiceContractError(
                "confirmed proposal did not return an activated monitor"
            )
        self._validate_inspection(inspection, monitor.id)
        return inspection

    async def reject(
        self,
        proposal_id: str,
        *,
        candidate_hash: str,
        actor_id: str,
        reason: str,
    ) -> MonitorConfirmation:
        """Reject the exact candidate without creating any monitor state."""

        proposal = await self._load_proposal(proposal_id)
        if candidate_hash != proposal.candidate_hash:
            raise MonitorProposalConflictError(
                proposal.id,
                "candidate_hash does not match the persisted proposal",
            )
        actor_id = _identity(actor_id, "confirmation actor_id")
        now = self._now()
        confirmation = MonitorConfirmation(
            id=self._id("monitor-confirmation"),
            agent_id=self._agent_id,
            proposal_id=proposal.id,
            decision=MonitorConfirmationDecision.REJECTED,
            candidate_hash=proposal.candidate_hash,
            actor_id=actor_id,
            reason=reason,
            decided_at=now,
        )
        inspection = await self._store.commit_monitor_confirmation(
            MonitorConfirmationCommit(
                proposal_id=proposal.id,
                confirmation=confirmation,
                event=self._event(
                    "monitor.rejected",
                    now=now,
                    monitor_id=proposal.intended_monitor_id,
                    operation_id=proposal.source_operation_id,
                    payload={
                        "candidate_hash": proposal.candidate_hash,
                        "proposal_id": proposal.id,
                    },
                ),
            )
        )
        if inspection is not None:
            raise MonitorServiceContractError(
                "rejected proposal unexpectedly activated a monitor"
            )
        stored_confirmation = await self._store.load_monitor_confirmation(
            self._agent_id,
            proposal.id,
        )
        if (
            not isinstance(stored_confirmation, MonitorConfirmation)
            or stored_confirmation.agent_id != self._agent_id
            or stored_confirmation.proposal_id != proposal.id
            or stored_confirmation.decision is not MonitorConfirmationDecision.REJECTED
            or stored_confirmation.candidate_hash != proposal.candidate_hash
        ):
            raise MonitorServiceContractError(
                "rejected proposal did not return its durable confirmation"
            )
        return stored_confirmation

    async def list(
        self,
        *,
        statuses: Sequence[MonitorStatus] | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> tuple[Monitor, ...]:
        """List bounded agent-scoped monitor projections."""

        if not isinstance(include_deleted, bool):
            raise TypeError("include_deleted must be a boolean")
        if statuses is None:
            selected: tuple[MonitorStatus, ...] = (
                MonitorStatus.ENABLED,
                MonitorStatus.PAUSED,
            )
            if include_deleted:
                selected += (MonitorStatus.DELETED,)
        else:
            if isinstance(statuses, (str, bytes)):
                raise TypeError("statuses must be a sequence of MonitorStatus values")
            selected = tuple(statuses)
            if not selected or any(
                not isinstance(status, MonitorStatus) for status in selected
            ):
                raise TypeError("statuses must contain MonitorStatus values")
            if len(selected) != len(set(selected)):
                raise ValueError("statuses cannot contain duplicates")
            if MonitorStatus.DELETED in selected and not include_deleted:
                raise ValueError("deleted monitors require include_deleted=True")
        values = tuple(
            await self._store.list_monitors(
                self._agent_id,
                statuses=selected,
                limit=_list_limit(limit),
            )
        )
        if len(values) > limit:
            raise MonitorServiceContractError("monitor store exceeded the list bound")
        if any(
            not isinstance(value, Monitor)
            or value.agent_id != self._agent_id
            or value.status not in selected
            for value in values
        ):
            raise MonitorServiceContractError(
                "monitor store returned an out-of-scope monitor"
            )
        if len({value.id for value in values}) != len(values):
            raise MonitorServiceContractError(
                "monitor store returned duplicate monitors"
            )
        return values

    async def list_proposals(self, *, limit: int = 100) -> tuple[MonitorProposal, ...]:
        values = tuple(
            await self._store.list_monitor_proposals(
                self._agent_id,
                limit=_list_limit(limit),
            )
        )
        if len(values) > limit or any(
            not isinstance(value, MonitorProposal) or value.agent_id != self._agent_id
            for value in values
        ):
            raise MonitorServiceContractError(
                "monitor store returned out-of-scope proposals"
            )
        return values

    async def inspect(self, monitor_id: str) -> MonitorInspection:
        monitor_id = _identity(monitor_id, "monitor_id")
        inspection = await self._store.inspect_monitor(self._agent_id, monitor_id)
        if inspection is None:
            raise MonitorNotFoundError(self._agent_id, monitor_id)
        self._validate_inspection(inspection, monitor_id)
        return inspection

    async def pause(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        return await self._transition(
            monitor_id,
            action=MonitorLifecycleAction.PAUSE,
            target_status=MonitorStatus.PAUSED,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def resume(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        return await self._transition(
            monitor_id,
            action=MonitorLifecycleAction.RESUME,
            target_status=MonitorStatus.ENABLED,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def delete(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        return await self._transition(
            monitor_id,
            action=MonitorLifecycleAction.DELETE,
            target_status=MonitorStatus.DELETED,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            operation_id=operation_id,
        )

    async def run_now(
        self,
        monitor_id: str,
        *,
        idempotency_key: str,
        holder_id: str | None = None,
        lease_seconds: float = 60.0,
    ) -> MonitorClaimResult:
        """Claim one manual occurrence; execution remains host/loop owned."""

        idempotency_key = _identity(
            idempotency_key,
            "run-now idempotency_key",
        )
        if holder_id is None:
            holder_id = self._id("monitor-run-now-holder")
        else:
            holder_id = _identity(holder_id, "run-now holder_id")
        if (
            not isinstance(lease_seconds, (int, float))
            or isinstance(lease_seconds, bool)
            or not math.isfinite(float(lease_seconds))
            or float(lease_seconds) <= 0
            or float(lease_seconds) > _MAX_LEASE_SECONDS
        ):
            raise ValueError(
                f"lease_seconds must be finite and from 0 through {_MAX_LEASE_SECONDS}"
            )
        inspection = await self.inspect(monitor_id)
        if inspection.monitor.status is MonitorStatus.DELETED:
            raise MonitorLifecycleError("deleted monitor cannot run now")
        now = self._now()
        existing = await self._store.load_monitor_claim_by_manual_key(
            self._agent_id,
            inspection.monitor.id,
            idempotency_key,
        )
        if existing is not None:
            self._validate_claim_result(
                existing,
                monitor_id=inspection.monitor.id,
                manual_key=idempotency_key,
            )
            reclaimable = (
                existing.run.status
                in {MonitorRunStatus.PENDING, MonitorRunStatus.WAITING}
                and existing.run.completed_at is None
                and existing.lease.expires_at <= now
            )
            if not reclaimable:
                return existing
            occurrence = existing.occurrence
            fencing_token = existing.lease.fencing_token + 1
            attempt = existing.run.attempt + 1
            operation_id = existing.run.operation_id
        else:
            occurrence_key = monitor_occurrence_key(
                agent_id=self._agent_id,
                monitor_id=inspection.monitor.id,
                monitor_version=inspection.monitor.current_version,
                kind=MonitorOccurrenceKind.RUN_NOW,
                scheduled_for=now,
                manual_key=idempotency_key,
            )
            occurrence = MonitorOccurrence(
                id=monitor_occurrence_id(occurrence_key),
                agent_id=self._agent_id,
                monitor_id=inspection.monitor.id,
                monitor_version=inspection.monitor.current_version,
                kind=MonitorOccurrenceKind.RUN_NOW,
                scheduled_for=now,
                occurrence_key=occurrence_key,
                trigger_id=monitor_trigger_id(occurrence_key),
                run_id=monitor_run_id(occurrence_key),
                created_at=now,
                manual_key=idempotency_key,
            )
            fencing_token = 1
            attempt = 1
            operation_id = None
        lease = MonitorTickLease(
            id=self._id("monitor-lease"),
            agent_id=self._agent_id,
            monitor_id=inspection.monitor.id,
            occurrence_id=occurrence.id,
            holder_id=holder_id,
            fencing_token=fencing_token,
            claimed_at=now,
            expires_at=now + timedelta(seconds=float(lease_seconds)),
        )
        run = MonitorRun(
            id=occurrence.run_id,
            agent_id=self._agent_id,
            monitor_id=inspection.monitor.id,
            occurrence_id=occurrence.id,
            trigger_id=occurrence.trigger_id,
            attempt=attempt,
            fencing_token=fencing_token,
            status=MonitorRunStatus.PENDING,
            started_at=now,
            operation_id=operation_id,
        )
        result = await self._store.claim_monitor_occurrence(
            MonitorOccurrenceClaim(
                occurrence=occurrence,
                lease=lease,
                run=run,
                event=self._event(
                    "monitor.claimed",
                    now=now,
                    monitor_id=inspection.monitor.id,
                    operation_id=None,
                    payload={
                        "fencing_token": fencing_token,
                        "kind": MonitorOccurrenceKind.RUN_NOW.value,
                        "occurrence_id": occurrence.id,
                        "operation_id": operation_id,
                        "run_id": run.id,
                        "trigger_id": occurrence.trigger_id,
                    },
                ),
            ),
            expected_monitor_revision=inspection.monitor.revision,
            expected_schedule_revision=inspection.schedule_state.revision,
            checked_at=now,
        )
        self._validate_claim_result(
            result,
            monitor_id=inspection.monitor.id,
            manual_key=idempotency_key,
        )
        return result

    async def _transition(
        self,
        monitor_id: str,
        *,
        action: MonitorLifecycleAction,
        target_status: MonitorStatus,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None,
    ) -> MonitorInspection:
        actor_id = _identity(actor_id, "lifecycle actor_id")
        idempotency_key = _identity(
            idempotency_key,
            "lifecycle idempotency_key",
        )
        operation_id = _optional_identity(operation_id, "lifecycle operation_id")
        inspection = await self.inspect(monitor_id)
        current = inspection.monitor
        if current.status is MonitorStatus.DELETED:
            if action is not MonitorLifecycleAction.DELETE:
                raise MonitorLifecycleError(
                    "deleted monitor cannot change lifecycle state"
                )
        now = self._now()
        updated = Monitor(
            id=current.id,
            agent_id=current.agent_id,
            status=target_status,
            current_version=current.current_version,
            revision=current.revision + 1,
            created_at=current.created_at,
            updated_at=now,
            paused_at=(
                current.paused_at or now
                if target_status is MonitorStatus.PAUSED
                else None
            ),
            deleted_at=(
                current.deleted_at or now
                if target_status is MonitorStatus.DELETED
                else None
            ),
        )
        state = inspection.schedule_state
        if action is MonitorLifecycleAction.RESUME:
            next_scheduled_at = inspection.versions[-1].definition.schedule.next_due_at(
                now
            )
        elif action is MonitorLifecycleAction.DELETE:
            next_scheduled_at = None
        else:
            next_scheduled_at = state.next_scheduled_at
        schedule_state = replace(
            state,
            revision=state.revision + 1,
            next_scheduled_at=next_scheduled_at,
            updated_at=now,
        )
        lifecycle = MonitorLifecycleRecord(
            id=self._id("monitor-lifecycle"),
            agent_id=self._agent_id,
            monitor_id=current.id,
            action=action,
            from_status=current.status,
            to_status=target_status,
            from_revision=current.revision,
            to_revision=updated.revision,
            monitor_version=current.current_version,
            actor_id=actor_id,
            reason=reason,
            idempotency_key=idempotency_key,
            occurred_at=now,
            operation_id=operation_id,
        )
        committed = await self._store.commit_monitor_lifecycle(
            MonitorLifecycleCommit(
                monitor=updated,
                lifecycle=lifecycle,
                schedule_state=schedule_state,
                event=self._event(
                    f"monitor.{action.value}",
                    now=now,
                    monitor_id=current.id,
                    operation_id=operation_id,
                    payload={
                        "from_revision": current.revision,
                        "from_status": current.status.value,
                        "idempotency_key": idempotency_key,
                        "to_revision": updated.revision,
                        "to_status": target_status.value,
                    },
                ),
            ),
            expected_revision=current.revision,
        )
        self._validate_inspection(committed, current.id)
        return committed

    async def _load_proposal(self, proposal_id: str) -> MonitorProposal:
        proposal_id = _identity(proposal_id, "proposal_id")
        proposal = await self._store.load_monitor_proposal(
            self._agent_id,
            proposal_id,
        )
        if proposal is None:
            raise MonitorProposalNotFoundError(self._agent_id, proposal_id)
        if not isinstance(proposal, MonitorProposal) or (
            proposal.agent_id != self._agent_id or proposal.id != proposal_id
        ):
            raise MonitorServiceContractError(
                "monitor store returned another proposal identity"
            )
        return proposal

    def _event(
        self,
        event_type: str,
        *,
        now: datetime,
        monitor_id: str,
        operation_id: str | None,
        payload: dict[str, object],
    ) -> RuntimeEvent:
        event_payload = dict(payload)
        if operation_id is not None:
            event_payload["source_operation_id"] = operation_id
        return RuntimeEvent(
            id=self._id("event"),
            type=event_type,
            agent_id=self._agent_id,
            operation_id=None,
            created_at=now,
            monitor_id=monitor_id,
            payload=event_payload,
        )

    def _id(self, prefix: str) -> str:
        return _identity(self._id_factory(prefix), f"{prefix} ID")

    def _now(self) -> datetime:
        value = self._clock()
        if (
            not isinstance(value, datetime)
            or value.tzinfo is None
            or value.utcoffset() is None
        ):
            raise ValueError(
                "monitor service clock must return a timezone-aware datetime"
            )
        return value.astimezone(timezone.utc)

    def _validate_proposal_replay(
        self,
        requested: MonitorProposal,
        stored: MonitorProposal,
    ) -> None:
        if not isinstance(stored, MonitorProposal):
            raise MonitorServiceContractError(
                "monitor store create_monitor_proposal returned an invalid record"
            )
        if (
            stored.agent_id != self._agent_id
            or stored.intended_monitor_id != requested.intended_monitor_id
            or stored.idempotency_key != requested.idempotency_key
            or stored.candidate_hash != requested.candidate_hash
        ):
            raise MonitorServiceContractError(
                "monitor store returned a divergent proposal replay"
            )

    def _validate_inspection(
        self,
        inspection: MonitorInspection,
        monitor_id: str,
    ) -> None:
        if not isinstance(inspection, MonitorInspection) or (
            inspection.monitor.agent_id != self._agent_id
            or inspection.monitor.id != monitor_id
        ):
            raise MonitorServiceContractError(
                "monitor store returned another monitor inspection"
            )

    def _validate_claim_result(
        self,
        result: MonitorClaimResult,
        *,
        monitor_id: str,
        manual_key: str,
    ) -> None:
        if not isinstance(result, MonitorClaimResult):
            raise MonitorServiceContractError(
                "monitor store returned an invalid occurrence claim"
            )
        if (
            result.occurrence.agent_id != self._agent_id
            or result.occurrence.monitor_id != monitor_id
            or result.occurrence.kind is not MonitorOccurrenceKind.RUN_NOW
            or result.occurrence.manual_key != manual_key
            or result.run.occurrence_id != result.occurrence.id
            or result.lease.occurrence_id != result.occurrence.id
        ):
            raise MonitorServiceContractError(
                "monitor store returned a divergent run-now claim"
            )


__all__ = [
    "MonitorLifecycleError",
    "MonitorService",
    "MonitorServiceContractError",
]
