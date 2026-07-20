"""One-shot monitor scheduling through the ordinary agent trigger path."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import math
from typing import Protocol
from uuid import uuid4

from .._json import FrozenJsonObject, canonical_json
from ..events.models import RuntimeEvent

# Canonical budget values are restricted here; loop execution remains host-owned.
from ..loop.models import LoopBudgets
from ..operations.checkpoints import OperationSnapshot
from ..operations.governance import DefaultPolicyProfile
from ..operations.models import AgentTrigger, OperationStatus, TriggerKind
from ..operations.store import OperationStore
from .models import (
    MonitorCheckpoint,
    MonitorDefinition,
    MonitorFinding,
    MonitorFindingSeverity,
    MonitorInspection,
    MonitorOccurrence,
    MonitorOccurrenceKind,
    MonitorRun,
    MonitorRunStatus,
    MonitorScheduleState,
    MonitorStatus,
    MonitorTickLease,
    MonitorTickLeaseGuard,
    advance_next_due_at,
    monitor_occurrence_id,
    monitor_occurrence_key,
    monitor_run_id,
    monitor_trigger_id,
)
from .store import (
    MonitorClaimResult,
    MonitorConflictError,
    MonitorOccurrenceClaim,
    MonitorOutcomeCommit,
    MonitorStore,
    MonitorTickClaimConflictError,
    StaleMonitorFenceError,
)

_MAX_LIMIT = 1_000
_MAX_LEASE_SECONDS = 300


def monitor_execution_settings(
    definition: MonitorDefinition,
    *,
    default_budgets: LoopBudgets,
    default_policy: DefaultPolicyProfile,
) -> tuple[LoopBudgets, DefaultPolicyProfile]:
    """Derive the restriction-only settings persisted on a monitor operation."""

    if not isinstance(definition, MonitorDefinition):
        raise TypeError("definition must be a MonitorDefinition")
    if not isinstance(default_budgets, LoopBudgets):
        raise TypeError("default_budgets must be LoopBudgets")
    if not isinstance(default_policy, DefaultPolicyProfile):
        raise TypeError("default_policy must be DefaultPolicyProfile")
    policy_overrides = dict(definition.policy_overrides)
    operation_template = dict(definition.operation_template)
    if policy_overrides not in ({}, {"mode": "read_only"}):
        raise ValueError("monitor policy overrides support only read_only mode")
    if operation_template not in ({}, {"domain": "data"}):
        raise ValueError("monitor operation template supports only the data domain")
    overrides = definition.budget_overrides
    requested = (
        ("max_turns", overrides.max_turns, default_budgets.max_turns),
        (
            "max_capability_calls",
            overrides.max_capability_calls,
            default_budgets.max_actions,
        ),
        (
            "max_wall_time_seconds",
            overrides.max_wall_time_seconds,
            default_budgets.max_wall_time_seconds,
        ),
    )
    if any(value is not None and value > limit for _, value, limit in requested):
        raised = next(
            name
            for name, value, limit in requested
            if value is not None and value > limit
        )
        raise ValueError(f"monitor {raised} may only restrict the agent default")
    budgets = replace(
        default_budgets,
        max_turns=(
            default_budgets.max_turns
            if overrides.max_turns is None
            else overrides.max_turns
        ),
        max_actions=(
            default_budgets.max_actions
            if overrides.max_capability_calls is None
            else overrides.max_capability_calls
        ),
        max_wall_time_seconds=(
            default_budgets.max_wall_time_seconds
            if overrides.max_wall_time_seconds is None
            else float(overrides.max_wall_time_seconds)
        ),
    )
    return budgets, DefaultPolicyProfile(
        id=default_policy.id,
        version=default_policy.version,
        allow_destructive=False,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _utc(value: datetime, name: str) -> datetime:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{name} must be timezone-aware")
    return value.astimezone(timezone.utc)


def _text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


class MonitorSchedulerContractError(RuntimeError):
    """A scheduler dependency returned state outside its durable contract."""


class MonitorTriggerRunner(Protocol):
    """Structural seam implemented by ``AgentLoop`` without importing it."""

    async def run(self, trigger: AgentTrigger) -> object: ...


@dataclass(frozen=True, slots=True)
class MonitorOutcomeProjection:
    """Execution-free interpretation of a successful operation's evidence."""

    matched: bool
    evidence_id: str | None = None
    cursor: Mapping[str, object] | None = None
    severity: MonitorFindingSeverity = MonitorFindingSeverity.INFO
    summary: str | None = None
    details: Mapping[str, object] = field(default_factory=dict)
    dedupe_key: str | None = None

    def __post_init__(self) -> None:
        cursor = (
            None if self.cursor is None else FrozenJsonObject.from_mapping(self.cursor)
        )
        details = FrozenJsonObject.from_mapping(self.details)
        if (self.matched or cursor is not None) and self.evidence_id is None:
            raise ValueError("match or cursor projection requires evidence_id")
        if self.matched:
            _text(self.summary or "", "projection summary")
            _text(self.dedupe_key or "", "projection dedupe_key")
        elif self.summary is not None or self.dedupe_key is not None or details:
            raise ValueError("unmatched projection cannot contain finding fields")
        if not isinstance(self.severity, MonitorFindingSeverity):
            raise TypeError("projection severity must be MonitorFindingSeverity")
        object.__setattr__(self, "cursor", cursor)
        object.__setattr__(self, "details", details)


class MonitorOutcomeProjector(Protocol):
    async def project(
        self,
        *,
        definition: MonitorDefinition,
        operation: OperationSnapshot,
        checkpoint: MonitorCheckpoint | None,
    ) -> MonitorOutcomeProjection: ...


@dataclass(frozen=True, slots=True)
class MonitorSchedulerResult:
    monitor_id: str
    occurrence_id: str
    claimed: bool
    reason: str
    run_status: MonitorRunStatus | None = None
    operation_id: str | None = None
    finding_id: str | None = None


class MonitorScheduler:
    """Claim and process one bounded, sequential due-monitor pass."""

    def __init__(
        self,
        *,
        agent_id: str,
        store: MonitorStore,
        operations: OperationStore,
        runner: MonitorTriggerRunner,
        projector: MonitorOutcomeProjector,
        holder_id: str,
        default_budgets: LoopBudgets = LoopBudgets(),
        default_policy: DefaultPolicyProfile = DefaultPolicyProfile(),
        lease_seconds: float = 300.0,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
    ) -> None:
        self._agent_id = _text(agent_id, "scheduler agent_id")
        self._holder_id = _text(holder_id, "scheduler holder_id")
        for owner, methods in (
            (
                store,
                (
                    "list_due_monitors",
                    "inspect_monitor",
                    "load_occurrence_by_trigger",
                    "claim_monitor_occurrence",
                    "commit_monitor_outcome",
                ),
            ),
            (operations, ("load_by_trigger",)),
            (runner, ("run",)),
            (projector, ("project",)),
        ):
            if any(not callable(getattr(owner, method, None)) for method in methods):
                raise TypeError("scheduler dependency does not implement its protocol")
        if (
            not isinstance(lease_seconds, (int, float))
            or isinstance(lease_seconds, bool)
            or not math.isfinite(float(lease_seconds))
            or not 0 < float(lease_seconds) <= _MAX_LEASE_SECONDS
        ):
            raise ValueError("lease_seconds must be finite, positive, and bounded")
        if not callable(clock) or not callable(id_factory):
            raise TypeError("clock and id_factory must be callable")
        self._store = store
        self._operations = operations
        self._runner = runner
        self._projector = projector
        if not isinstance(default_budgets, LoopBudgets):
            raise TypeError("default_budgets must be LoopBudgets")
        if not isinstance(default_policy, DefaultPolicyProfile):
            raise TypeError("default_policy must be DefaultPolicyProfile")
        self._default_budgets = default_budgets
        self._default_policy = default_policy
        self._lease_seconds = float(lease_seconds)
        self._clock = clock
        self._id_factory = id_factory

    async def run_due(
        self,
        now: datetime,
        *,
        limit: int = 100,
    ) -> tuple[MonitorSchedulerResult, ...]:
        """Run once; recurring cadence remains exclusively host-owned."""

        now = _utc(now, "scheduler now")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= _MAX_LIMIT
        ):
            raise ValueError(f"limit must be from 1 through {_MAX_LIMIT}")
        due = tuple(
            await self._store.list_due_monitors(
                self._agent_id,
                now=now,
                limit=limit,
            )
        )
        if len(due) > limit:
            raise MonitorSchedulerContractError("monitor store exceeded due limit")
        results: list[MonitorSchedulerResult] = []
        for inspection in due:
            self._validate_due(inspection)
            occurrence = self._occurrence(inspection, now)
            requested = self._claim(inspection, occurrence, now)
            try:
                claim = await self._store.claim_monitor_occurrence(
                    requested,
                    expected_monitor_revision=inspection.monitor.revision,
                    expected_schedule_revision=inspection.schedule_state.revision,
                    checked_at=now,
                )
            except (
                MonitorConflictError,
                MonitorTickClaimConflictError,
                StaleMonitorFenceError,
            ):
                results.append(
                    MonitorSchedulerResult(
                        inspection.monitor.id,
                        occurrence.id,
                        False,
                        "claim_conflict",
                    )
                )
                continue
            if (
                not isinstance(claim, MonitorClaimResult)
                or claim.occurrence != occurrence
                or claim.run.status is not MonitorRunStatus.PENDING
            ):
                raise MonitorSchedulerContractError("store returned invalid claim")
            results.append(await self._run_claimed(inspection, claim))
        return tuple(results)

    async def run_claimed(
        self,
        claim: MonitorClaimResult,
    ) -> MonitorSchedulerResult:
        """Execute one already-durable run-now claim through the normal loop."""

        if not isinstance(claim, MonitorClaimResult):
            raise TypeError("claim must be a MonitorClaimResult")
        occurrence = claim.occurrence
        if occurrence.agent_id != self._agent_id:
            raise ValueError("monitor claim belongs to another agent")
        inspection = await self._store.inspect_monitor(
            self._agent_id,
            occurrence.monitor_id,
        )
        if inspection is None:
            raise MonitorSchedulerContractError("claimed monitor no longer exists")
        self._validate_claim(inspection, claim)
        if claim.run.status is not MonitorRunStatus.PENDING:
            return MonitorSchedulerResult(
                occurrence.monitor_id,
                occurrence.id,
                False,
                f"already_{claim.run.status.value}",
                claim.run.status,
                claim.run.operation_id,
            )
        return await self._run_claimed(inspection, claim)

    async def resume_trigger(
        self,
        trigger_id: str,
    ) -> MonitorSchedulerResult:
        """Reclaim and resume one waiting monitor occurrence after a wakeup."""

        trigger_id = _text(trigger_id, "monitor trigger_id")
        occurrence = await self._store.load_occurrence_by_trigger(
            self._agent_id,
            trigger_id,
        )
        if occurrence is None:
            raise MonitorSchedulerContractError("monitor trigger has no occurrence")
        inspection = await self._store.inspect_monitor(
            self._agent_id,
            occurrence.monitor_id,
        )
        if inspection is None:
            raise MonitorSchedulerContractError("monitor trigger has no monitor")
        self._validate_due_identity(inspection)
        prior_run = next(
            (item for item in inspection.runs if item.id == occurrence.run_id),
            None,
        )
        if prior_run is None:
            raise MonitorSchedulerContractError("monitor occurrence has no run")
        if prior_run.status in {
            MonitorRunStatus.SUCCEEDED,
            MonitorRunStatus.FAILED,
            MonitorRunStatus.CANCELLED,
            MonitorRunStatus.SKIPPED,
        }:
            return MonitorSchedulerResult(
                occurrence.monitor_id,
                occurrence.id,
                False,
                f"already_{prior_run.status.value}",
                prior_run.status,
                prior_run.operation_id,
            )
        now = _utc(self._clock(), "scheduler clock")
        requested = self._claim(inspection, occurrence, now)
        claim = await self._store.claim_monitor_occurrence(
            requested,
            expected_monitor_revision=inspection.monitor.revision,
            expected_schedule_revision=inspection.schedule_state.revision,
            checked_at=now,
        )
        if not isinstance(claim, MonitorClaimResult):
            raise MonitorSchedulerContractError("store returned invalid reclaim")
        claimed_inspection = await self._store.inspect_monitor(
            self._agent_id,
            occurrence.monitor_id,
        )
        if claimed_inspection is None:
            raise MonitorSchedulerContractError("reclaimed monitor disappeared")
        self._validate_claim(claimed_inspection, claim)
        return await self._run_claimed(claimed_inspection, claim)

    async def _run_claimed(
        self,
        inspection: MonitorInspection,
        claim: MonitorClaimResult,
    ) -> MonitorSchedulerResult:
        definition = self._definition_for_occurrence(
            inspection,
            claim.occurrence,
        )
        checkpoint = self._checkpoint(inspection)
        trigger = self._trigger(
            definition,
            claim.occurrence,
            checkpoint,
        )
        await self._runner.run(trigger)
        versioned = await self._operations.load_by_trigger(trigger.id)
        if versioned is None:
            raise MonitorSchedulerContractError("runner created no durable operation")
        snapshot = versioned.snapshot
        if snapshot.trigger != trigger or snapshot.operation.agent_id != self._agent_id:
            raise MonitorSchedulerContractError(
                "operation does not match monitor trigger"
            )
        expected_budgets, _ = monitor_execution_settings(
            definition,
            default_budgets=self._default_budgets,
            default_policy=self._default_policy,
        )
        if snapshot.budgets != expected_budgets:
            raise MonitorSchedulerContractError(
                "operation does not retain the effective monitor budgets"
            )
        finished_at = _utc(self._clock(), "scheduler clock")
        commit = await self._outcome(
            inspection,
            claim,
            snapshot,
            checkpoint,
            definition,
            finished_at,
        )
        outcome = await self._store.commit_monitor_outcome(
            commit,
            expected_monitor_revision=inspection.monitor.revision,
            expected_schedule_revision=inspection.schedule_state.revision,
            checked_at=finished_at,
        )
        return MonitorSchedulerResult(
            inspection.monitor.id,
            claim.occurrence.id,
            True,
            outcome.run.status.value,
            outcome.run.status,
            outcome.run.operation_id,
            None if outcome.finding is None else outcome.finding.id,
        )

    async def _outcome(
        self,
        inspection: MonitorInspection,
        claim: MonitorClaimResult,
        snapshot: OperationSnapshot,
        checkpoint: MonitorCheckpoint | None,
        definition: MonitorDefinition,
        finished_at: datetime,
    ) -> MonitorOutcomeCommit:
        operation_status = snapshot.operation.status
        projection: MonitorOutcomeProjection | None = None
        if operation_status is OperationStatus.SUCCEEDED:
            projection = await self._projector.project(
                definition=definition,
                operation=snapshot,
                checkpoint=checkpoint,
            )
            if not isinstance(projection, MonitorOutcomeProjection):
                raise MonitorSchedulerContractError("projector returned invalid result")
            self._validate_evidence(projection, snapshot)
            run_status = MonitorRunStatus.SUCCEEDED
        elif operation_status in {
            OperationStatus.WAITING_FOR_APPROVAL,
            OperationStatus.WAITING_FOR_INPUT,
        }:
            run_status = MonitorRunStatus.WAITING
        elif operation_status is OperationStatus.CANCELLED:
            run_status = MonitorRunStatus.CANCELLED
        elif operation_status in {OperationStatus.FAILED, OperationStatus.INTERRUPTED}:
            run_status = MonitorRunStatus.FAILED
        else:
            raise MonitorSchedulerContractError(
                f"runner stopped at non-waiting status {operation_status.value}"
            )
        terminal = run_status is not MonitorRunStatus.WAITING
        failure = None
        if run_status in {MonitorRunStatus.FAILED, MonitorRunStatus.CANCELLED}:
            failure = (snapshot.operation.terminal_reason or run_status.value)[:2_000]
        run = replace(
            claim.run,
            status=run_status,
            operation_id=snapshot.operation.id,
            completed_at=finished_at if terminal else None,
            failure_reason=failure,
        )
        released = replace(
            claim.lease,
            released_at=finished_at,
            release_reason=run_status.value,
        )
        finding = self._finding(
            inspection, claim.occurrence, run, projection, finished_at
        )
        monitor_checkpoint = self._new_checkpoint(
            inspection, run, projection, finished_at
        )
        events = [
            self._event(
                f"monitor.{run_status.value}",
                inspection.monitor.id,
                finished_at,
                {
                    "fencing_token": claim.lease.fencing_token,
                    "occurrence_id": claim.occurrence.id,
                    "operation_id": snapshot.operation.id,
                    "run_id": run.id,
                    "trigger_id": claim.occurrence.trigger_id,
                },
            )
        ]
        if finding is not None:
            events.append(
                self._event(
                    "monitor.finding_recorded",
                    inspection.monitor.id,
                    finished_at,
                    {
                        "dedupe_key": finding.dedupe_key,
                        "evidence_id": finding.evidence_id,
                        "finding_id": finding.id,
                        "operation_id": snapshot.operation.id,
                    },
                )
            )
        return MonitorOutcomeCommit(
            guard=MonitorTickLeaseGuard(
                self._agent_id,
                inspection.monitor.id,
                claim.occurrence.id,
                claim.lease.holder_id,
                claim.lease.fencing_token,
            ),
            run=run,
            released_lease=released,
            schedule_state=self._next_state(
                inspection,
                claim.occurrence,
                run,
                projection,
                finished_at,
            ),
            events=tuple(events),
            checkpoint=monitor_checkpoint,
            finding=finding,
        )

    def _occurrence(
        self,
        inspection: MonitorInspection,
        now: datetime,
    ) -> MonitorOccurrence:
        scheduled_for = inspection.schedule_state.next_scheduled_at
        if scheduled_for is None or scheduled_for > now:
            raise MonitorSchedulerContractError("due monitor has no due frontier")
        key = monitor_occurrence_key(
            agent_id=self._agent_id,
            monitor_id=inspection.monitor.id,
            monitor_version=inspection.monitor.current_version,
            kind=MonitorOccurrenceKind.SCHEDULED,
            scheduled_for=scheduled_for,
        )
        occurrence_id = monitor_occurrence_id(key)
        existing = next(
            (item for item in inspection.occurrences if item.id == occurrence_id),
            None,
        )
        if existing is not None:
            return existing
        return MonitorOccurrence(
            occurrence_id,
            self._agent_id,
            inspection.monitor.id,
            inspection.monitor.current_version,
            MonitorOccurrenceKind.SCHEDULED,
            scheduled_for,
            key,
            monitor_trigger_id(key),
            monitor_run_id(key),
            scheduled_for,
        )

    def _claim(
        self,
        inspection: MonitorInspection,
        occurrence: MonitorOccurrence,
        now: datetime,
    ) -> MonitorOccurrenceClaim:
        leases = [
            item for item in inspection.leases if item.occurrence_id == occurrence.id
        ]
        latest = max(leases, key=lambda item: item.fencing_token, default=None)
        prior_run = next(
            (item for item in inspection.runs if item.id == occurrence.run_id),
            None,
        )
        fence = 1 if latest is None else latest.fencing_token + 1
        attempt = 1 if prior_run is None else prior_run.attempt + 1
        lease = MonitorTickLease(
            self._id("monitor-lease"),
            self._agent_id,
            inspection.monitor.id,
            occurrence.id,
            self._holder_id,
            fence,
            now,
            now + timedelta(seconds=self._lease_seconds),
        )
        run = MonitorRun(
            occurrence.run_id,
            self._agent_id,
            inspection.monitor.id,
            occurrence.id,
            occurrence.trigger_id,
            attempt,
            fence,
            MonitorRunStatus.PENDING,
            now,
            None if prior_run is None else prior_run.operation_id,
        )
        return MonitorOccurrenceClaim(
            occurrence,
            lease,
            run,
            self._event(
                "monitor.claimed",
                inspection.monitor.id,
                now,
                {
                    "fencing_token": fence,
                    "kind": occurrence.kind.value,
                    "occurrence_id": occurrence.id,
                    "operation_id": run.operation_id,
                    "run_id": run.id,
                    "trigger_id": occurrence.trigger_id,
                },
            ),
        )

    def _trigger(
        self,
        definition: MonitorDefinition,
        occurrence: MonitorOccurrence,
        checkpoint: MonitorCheckpoint | None,
    ) -> AgentTrigger:
        condition = definition.condition
        budget = definition.budget_overrides
        effective_budgets, effective_policy = monitor_execution_settings(
            definition,
            default_budgets=self._default_budgets,
            default_policy=self._default_policy,
        )
        return AgentTrigger(
            id=occurrence.trigger_id,
            agent_id=self._agent_id,
            kind=TriggerKind.MONITOR,
            source_id=occurrence.monitor_id,
            session_id=None,
            created_at=occurrence.created_at,
            payload={
                "message": definition.objective,
                "monitor_id": occurrence.monitor_id,
                "monitor_version": occurrence.monitor_version,
                "monitor_definition_hash": definition.content_hash,
                "monitor_occurrence_id": occurrence.id,
                "monitor_run_id": occurrence.run_id,
                "monitor_kind": occurrence.kind.value,
                "monitor_scheduled_for": occurrence.scheduled_for.isoformat(),
                "monitor_scope": {
                    "source_ids": list(definition.scope.source_ids),
                    "resource_ids": list(definition.scope.resource_ids),
                },
                "monitor_condition": {
                    "kind": condition.kind.value,
                    "expression": condition.expression,
                    "configuration": condition.configuration,
                },
                "monitor_budget_overrides": {
                    "max_turns": budget.max_turns,
                    "max_capability_calls": budget.max_capability_calls,
                    "max_wall_time_seconds": budget.max_wall_time_seconds,
                },
                "monitor_effective_budgets": {
                    "max_actions": effective_budgets.max_actions,
                    "max_estimated_cost_usd": (
                        None
                        if effective_budgets.max_estimated_cost_usd is None
                        else str(effective_budgets.max_estimated_cost_usd)
                    ),
                    "max_identical_failures": effective_budgets.max_identical_failures,
                    "max_observation_characters": effective_budgets.max_observation_characters,
                    "max_repairs": effective_budgets.max_repairs,
                    "max_total_tokens": effective_budgets.max_total_tokens,
                    "max_turns": effective_budgets.max_turns,
                    "max_wall_time_seconds": effective_budgets.max_wall_time_seconds,
                    "task_timeout_seconds": effective_budgets.task_timeout_seconds,
                },
                "monitor_effective_policy": {
                    "allow_destructive": effective_policy.allow_destructive,
                    "fingerprint": effective_policy.fingerprint,
                    "id": effective_policy.id,
                    "version": effective_policy.version,
                },
                "monitor_policy_overrides": definition.policy_overrides,
                "monitor_operation_template": definition.operation_template,
                "monitor_checkpoint": (
                    None
                    if checkpoint is None
                    else {
                        "version": checkpoint.version,
                        "cursor": checkpoint.cursor,
                        "cursor_hash": checkpoint.cursor_hash,
                    }
                ),
            },
        )

    def _next_state(
        self,
        inspection: MonitorInspection,
        occurrence: MonitorOccurrence,
        run: MonitorRun,
        projection: MonitorOutcomeProjection | None,
        now: datetime,
    ) -> MonitorScheduleState:
        state = inspection.schedule_state
        definition = inspection.versions[-1].definition
        matched = projection is not None and projection.matched
        if run.status is MonitorRunStatus.WAITING:
            next_due = state.next_scheduled_at
            failures, matches = state.consecutive_failures, state.consecutive_matches
            backoff, cooldown = state.backoff_until, state.cooldown_until
        else:
            next_due = (
                state.next_scheduled_at
                if occurrence.kind is MonitorOccurrenceKind.RUN_NOW
                else advance_next_due_at(
                    definition.schedule,
                    completed_at=now,
                    catch_up=definition.timing.catch_up,
                )
            )
            if run.status is MonitorRunStatus.SUCCEEDED:
                failures = 0
                matches = state.consecutive_matches + 1 if matched else 0
                backoff = None
                cooldown = (
                    now + timedelta(seconds=definition.timing.cooldown_seconds)
                    if matched and definition.timing.cooldown_seconds
                    else None
                )
            elif run.status is MonitorRunStatus.FAILED:
                failures, matches, cooldown = state.consecutive_failures + 1, 0, None
                backoff = now + timedelta(
                    seconds=self._backoff(definition, state.consecutive_failures)
                )
            else:
                failures, matches, backoff, cooldown = 0, 0, None, None
        return replace(
            state,
            revision=state.revision + 1,
            next_scheduled_at=next_due,
            updated_at=now,
            last_scheduled_at=occurrence.scheduled_for,
            cooldown_until=cooldown,
            backoff_until=backoff,
            consecutive_failures=failures,
            consecutive_matches=matches,
            checkpoint_version=state.checkpoint_version
            + int(projection is not None and projection.cursor is not None),
            last_occurrence_id=occurrence.id,
            last_run_id=run.id,
            last_operation_id=run.operation_id,
        )

    @staticmethod
    def _backoff(definition: MonitorDefinition, failures: int) -> int:
        timing = definition.timing
        delay = timing.initial_backoff_seconds
        for _ in range(failures):
            delay = min(
                timing.max_backoff_seconds,
                math.ceil(delay * timing.backoff_multiplier),
            )
            if delay == timing.max_backoff_seconds:
                break
        return delay

    def _new_checkpoint(
        self,
        inspection: MonitorInspection,
        run: MonitorRun,
        projection: MonitorOutcomeProjection | None,
        now: datetime,
    ) -> MonitorCheckpoint | None:
        if projection is None or projection.cursor is None:
            return None
        digest = (
            "sha256:"
            + sha256(canonical_json(projection.cursor).encode("utf-8")).hexdigest()
        )
        version = inspection.schedule_state.checkpoint_version + 1
        return MonitorCheckpoint(
            self._id("monitor-checkpoint"),
            self._agent_id,
            inspection.monitor.id,
            version,
            run.id,
            projection.cursor,
            digest,
            now,
            None if version == 1 else version - 1,
        )

    def _finding(
        self,
        inspection: MonitorInspection,
        occurrence: MonitorOccurrence,
        run: MonitorRun,
        projection: MonitorOutcomeProjection | None,
        now: datetime,
    ) -> MonitorFinding | None:
        if projection is None or not projection.matched:
            return None
        if (
            projection.evidence_id is None
            or projection.summary is None
            or projection.dedupe_key is None
            or run.operation_id is None
        ):
            raise MonitorSchedulerContractError(
                "matched projection is missing finding correlation"
            )
        return MonitorFinding(
            self._id("monitor-finding"),
            self._agent_id,
            inspection.monitor.id,
            occurrence.id,
            run.id,
            run.operation_id,
            projection.evidence_id,
            projection.severity,
            projection.summary,
            projection.details,
            projection.dedupe_key,
            now,
        )

    @staticmethod
    def _validate_evidence(
        projection: MonitorOutcomeProjection,
        snapshot: OperationSnapshot,
    ) -> None:
        if projection.evidence_id is None:
            return
        evidence = next(
            (item for item in snapshot.evidence if item.id == projection.evidence_id),
            None,
        )
        if (
            evidence is None
            or not evidence.accepted
            or evidence.operation_id != snapshot.operation.id
        ):
            raise MonitorSchedulerContractError(
                "projection evidence is not accepted by its operation"
            )

    @staticmethod
    def _checkpoint(inspection: MonitorInspection) -> MonitorCheckpoint | None:
        version = inspection.schedule_state.checkpoint_version
        if version == 0:
            return None
        checkpoint = next(
            (item for item in inspection.checkpoints if item.version == version),
            None,
        )
        if checkpoint is None:
            raise MonitorSchedulerContractError("schedule checkpoint is missing")
        return checkpoint

    def _validate_due(self, inspection: MonitorInspection) -> None:
        self._validate_due_identity(inspection)
        if inspection.monitor.status is not MonitorStatus.ENABLED:
            raise MonitorSchedulerContractError("store returned invalid due monitor")

    def _validate_due_identity(self, inspection: MonitorInspection) -> None:
        if (
            not isinstance(inspection, MonitorInspection)
            or inspection.monitor.agent_id != self._agent_id
        ):
            raise MonitorSchedulerContractError("store returned invalid monitor")

    def _validate_claim(
        self,
        inspection: MonitorInspection,
        claim: MonitorClaimResult,
    ) -> None:
        self._validate_due_identity(inspection)
        if (
            claim.occurrence.monitor_id != inspection.monitor.id
            or claim.occurrence not in inspection.occurrences
            or claim.lease not in inspection.leases
            or claim.run not in inspection.runs
            or claim.run.id != claim.occurrence.run_id
            or claim.lease.occurrence_id != claim.occurrence.id
        ):
            raise MonitorSchedulerContractError(
                "claim does not match durable monitor state"
            )
        self._definition_for_occurrence(inspection, claim.occurrence)

    @staticmethod
    def _definition_for_occurrence(
        inspection: MonitorInspection,
        occurrence: MonitorOccurrence,
    ) -> MonitorDefinition:
        version = next(
            (
                item
                for item in inspection.versions
                if item.version == occurrence.monitor_version
            ),
            None,
        )
        if version is None:
            raise MonitorSchedulerContractError(
                "monitor occurrence has no retained definition version"
            )
        return version.definition

    def _event(
        self,
        event_type: str,
        monitor_id: str,
        now: datetime,
        payload: Mapping[str, object],
    ) -> RuntimeEvent:
        return RuntimeEvent(
            self._id("event"),
            event_type,
            self._agent_id,
            None,
            now,
            monitor_id=monitor_id,
            payload=payload,
        )

    def _id(self, prefix: str) -> str:
        return _text(self._id_factory(prefix), f"{prefix} ID")


__all__ = [
    "MonitorOutcomeProjection",
    "MonitorOutcomeProjector",
    "MonitorScheduler",
    "MonitorSchedulerContractError",
    "MonitorSchedulerResult",
    "MonitorTriggerRunner",
]
