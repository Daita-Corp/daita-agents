from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path

import pytest

from daita._json import canonical_json
from daita.monitors import (
    CatchUpPolicy,
    CronSchedule,
    IntervalSchedule,
    Monitor,
    MonitorBudgetOverrides,
    MonitorCheckpoint,
    MonitorCondition,
    MonitorConditionKind,
    MonitorConfirmation,
    MonitorConfirmationDecision,
    MonitorDefinition,
    MonitorFinding,
    MonitorFindingSeverity,
    MonitorInspection,
    MonitorLifecycleAction,
    MonitorLifecycleRecord,
    MonitorOccurrence,
    MonitorOccurrenceKind,
    MonitorProposal,
    MonitorRun,
    MonitorRunStatus,
    MonitorScheduleState,
    MonitorScope,
    MonitorStatus,
    MonitorTickLease,
    MonitorTickLeaseGuard,
    MonitorTimingPolicy,
    MonitorVersion,
    advance_next_due_at,
    monitor_occurrence_id,
    monitor_occurrence_key,
    monitor_run_id,
    monitor_trigger_id,
)

UTC = timezone.utc
NOW = datetime(2026, 7, 18, 12, 0, tzinfo=UTC)


def _definition(**changes: object) -> MonitorDefinition:
    values: dict[str, object] = {
        "name": "Daily quality check",
        "objective": "Compare the latest export with the canonical customer table.",
        "scope": MonitorScope(
            source_ids=("source-primary",),
            resource_ids=("resource-customers",),
        ),
        "schedule": CronSchedule("0 8 * * 1-5"),
        "condition": MonitorCondition(
            kind=MonitorConditionKind.THRESHOLD,
            expression="rows.0.discrepancy_percent",
            configuration={"operator": "gt", "value": 2},
        ),
        "budget_overrides": MonitorBudgetOverrides(
            max_turns=4,
            max_capability_calls=8,
            max_wall_time_seconds=60,
        ),
        "timing": MonitorTimingPolicy(cooldown_seconds=300),
        "policy_overrides": {"mode": "read_only"},
        "operation_template": {"domain": "data"},
    }
    values.update(changes)
    return MonitorDefinition(**values)  # type: ignore[arg-type]


def _occurrence(
    *,
    kind: MonitorOccurrenceKind = MonitorOccurrenceKind.SCHEDULED,
    manual_key: str | None = None,
) -> MonitorOccurrence:
    key = monitor_occurrence_key(
        agent_id="agent-a",
        monitor_id="monitor-a",
        monitor_version=1,
        kind=kind,
        scheduled_for=NOW,
        manual_key=manual_key,
    )
    return MonitorOccurrence(
        id=monitor_occurrence_id(key),
        agent_id="agent-a",
        monitor_id="monitor-a",
        monitor_version=1,
        kind=kind,
        scheduled_for=NOW,
        occurrence_key=key,
        trigger_id=monitor_trigger_id(key),
        run_id=monitor_run_id(key),
        created_at=NOW,
        manual_key=manual_key,
    )


def _cursor_hash(cursor: dict[str, object]) -> str:
    return "sha256:" + sha256(canonical_json(cursor).encode()).hexdigest()


def test_interval_and_cron_next_due_are_strict_deterministic_and_utc() -> None:
    interval = IntervalSchedule(interval_seconds=900, anchor_at=NOW)
    assert interval.next_due_at(NOW - timedelta(seconds=1)) == NOW
    assert interval.next_due_at(NOW) == NOW + timedelta(minutes=15)
    assert interval.next_due_at(NOW + timedelta(minutes=16)) == NOW + timedelta(
        minutes=30
    )

    weekdays = CronSchedule("*/15 8-9 * * 1-5")
    friday = datetime(2026, 7, 17, 9, 59, 59, tzinfo=UTC)
    assert weekdays.next_due_at(friday) == datetime(2026, 7, 20, 8, 0, tzinfo=UTC)
    assert weekdays.next_due_at(friday).tzinfo is UTC

    # Cron follows the conventional day-of-month OR day-of-week rule.
    day_or_weekday = CronSchedule("0 8 20 * 2")
    assert day_or_weekday.next_due_at(NOW) == datetime(2026, 7, 20, 8, 0, tzinfo=UTC)


def test_catch_up_once_advances_from_completion_instead_of_replaying_backlog() -> None:
    schedule = IntervalSchedule(interval_seconds=60, anchor_at=NOW)
    resumed_at = NOW + timedelta(hours=8, seconds=5)
    assert advance_next_due_at(
        schedule,
        completed_at=resumed_at,
        catch_up=CatchUpPolicy.ONCE,
    ) == NOW + timedelta(hours=8, minutes=1)


@pytest.mark.parametrize(
    ("factory", "message"),
    [
        (lambda: IntervalSchedule(0, NOW), "interval_seconds"),
        (
            lambda: IntervalSchedule(60, datetime(2026, 7, 18, 12, 0)),
            "timezone-aware",
        ),
        (lambda: CronSchedule("0 8 * *"), "five fields"),
        (lambda: CronSchedule("60 8 * * *"), "cron value"),
        (lambda: CronSchedule("0 8 * * *", timezone_name="America/Chicago"), "UTC"),
        (
            lambda: MonitorScope(resource_ids=("resource-a",)),
            "requires at least one source_id",
        ),
        (
            lambda: MonitorScope(source_ids=("source-b", "source-a")),
            "unique and sorted",
        ),
        (
            lambda: MonitorCondition(kind=MonitorConditionKind.EXPRESSION),
            "unsupported",
        ),
        (lambda: MonitorBudgetOverrides(max_turns=0), "max_turns"),
        (lambda: _definition(name="bad/name"), "human-readable"),
        (lambda: _definition(objective=" "), "objective"),
    ],
)
def test_strict_schedule_scope_condition_and_definition_invalids(
    factory: object,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        factory()  # type: ignore[operator]


def test_definition_hash_is_canonical_immutable_and_sensitive_to_semantics() -> None:
    limits = [1, 2]
    policy: dict[str, object] = {"mode": "read_only", "limits": limits}
    first = _definition(policy_overrides=policy)
    policy["mode"] = "write"
    limits.append(3)
    second = _definition(policy_overrides={"limits": [1, 2], "mode": "read_only"})

    assert first.content_hash == second.content_hash
    assert first.policy_overrides["mode"] == "read_only"
    assert (
        _definition(objective="A different objective.").content_hash
        != first.content_hash
    )
    with pytest.raises(FrozenInstanceError):
        first.objective = "mutated"  # type: ignore[misc]


def test_proposal_and_confirmation_bind_the_exact_candidate_hash() -> None:
    definition = _definition()
    proposal = MonitorProposal(
        id="proposal-a",
        agent_id="agent-a",
        intended_monitor_id="monitor-a",
        idempotency_key="proposal-key-a",
        candidate=definition,
        candidate_hash=definition.content_hash,
        created_at=NOW,
    )
    confirmation = MonitorConfirmation(
        id="confirmation-a",
        agent_id="agent-a",
        proposal_id=proposal.id,
        decision=MonitorConfirmationDecision.CONFIRMED,
        candidate_hash=proposal.candidate_hash,
        actor_id="user-a",
        reason="The schedule, scope, condition, and budgets are correct.",
        decided_at=NOW,
        resulting_monitor_id="monitor-a",
        resulting_version_id="monitor-version-a",
    )
    assert confirmation.candidate_hash == proposal.candidate_hash

    with pytest.raises(ValueError, match="does not match"):
        MonitorProposal(
            id="proposal-b",
            agent_id="agent-a",
            intended_monitor_id="monitor-a",
            idempotency_key="proposal-key-b",
            candidate=definition,
            candidate_hash="sha256:" + ("0" * 64),
            created_at=NOW,
        )
    with pytest.raises(ValueError, match="requires monitor and version IDs"):
        MonitorConfirmation(
            id="confirmation-b",
            agent_id="agent-a",
            proposal_id=proposal.id,
            decision=MonitorConfirmationDecision.CONFIRMED,
            candidate_hash=proposal.candidate_hash,
            actor_id="user-a",
            reason="Confirm.",
            decided_at=NOW,
        )


def test_occurrence_identity_is_stable_and_kind_scoped() -> None:
    scheduled = _occurrence()
    replay = _occurrence()
    run_now = _occurrence(
        kind=MonitorOccurrenceKind.RUN_NOW,
        manual_key="request-a",
    )

    assert replay == scheduled
    assert scheduled.id != run_now.id
    assert scheduled.trigger_id != run_now.trigger_id
    assert scheduled.run_id != run_now.run_id
    with pytest.raises(ValueError, match="manual_key"):
        _occurrence(kind=MonitorOccurrenceKind.RUN_NOW)
    with pytest.raises(ValueError, match="does not match"):
        MonitorOccurrence(
            id=scheduled.id,
            agent_id=scheduled.agent_id,
            monitor_id=scheduled.monitor_id,
            monitor_version=scheduled.monitor_version,
            kind=scheduled.kind,
            scheduled_for=scheduled.scheduled_for,
            occurrence_key=scheduled.occurrence_key,
            trigger_id="monitor-trigger-wrong",
            run_id=scheduled.run_id,
            created_at=NOW,
        )


def test_lease_run_finding_checkpoint_and_inspection_form_one_auditable_chain() -> None:
    definition = _definition()
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
        idempotency_key="activate-a",
        occurred_at=NOW,
    )
    occurrence = _occurrence()
    lease = MonitorTickLease(
        id="lease-a",
        agent_id="agent-a",
        monitor_id=monitor.id,
        occurrence_id=occurrence.id,
        holder_id="host-a",
        fencing_token=1,
        claimed_at=NOW,
        expires_at=NOW + timedelta(minutes=1),
    )
    guard = MonitorTickLeaseGuard(
        agent_id=lease.agent_id,
        monitor_id=lease.monitor_id,
        occurrence_id=lease.occurrence_id,
        holder_id=lease.holder_id,
        fencing_token=lease.fencing_token,
    )
    run = MonitorRun(
        id=occurrence.run_id,
        agent_id="agent-a",
        monitor_id=monitor.id,
        occurrence_id=occurrence.id,
        trigger_id=occurrence.trigger_id,
        attempt=1,
        fencing_token=guard.fencing_token,
        status=MonitorRunStatus.SUCCEEDED,
        started_at=NOW,
        operation_id="operation-a",
        completed_at=NOW + timedelta(seconds=10),
    )
    completed_at = run.completed_at
    assert completed_at is not None
    finding = MonitorFinding(
        id="finding-a",
        agent_id="agent-a",
        monitor_id=monitor.id,
        occurrence_id=occurrence.id,
        run_id=run.id,
        operation_id="operation-a",
        evidence_id="evidence-a",
        severity=MonitorFindingSeverity.WARNING,
        summary="Customer discrepancy exceeded two percent.",
        details={"discrepancy_percent": 2.4},
        dedupe_key="finding-dedupe-a",
        created_at=completed_at,
    )
    cursor: dict[str, object] = {"latest_revision": "revision-a"}
    checkpoint = MonitorCheckpoint(
        id="checkpoint-a",
        agent_id="agent-a",
        monitor_id=monitor.id,
        version=1,
        run_id=run.id,
        cursor=cursor,
        cursor_hash=_cursor_hash(cursor),
        created_at=completed_at,
    )
    state = MonitorScheduleState(
        agent_id="agent-a",
        monitor_id=monitor.id,
        revision=1,
        next_scheduled_at=NOW + timedelta(days=1),
        updated_at=completed_at,
        last_scheduled_at=NOW,
        checkpoint_version=1,
        last_occurrence_id=occurrence.id,
        last_run_id=run.id,
        last_operation_id="operation-a",
    )
    inspection = MonitorInspection(
        monitor=monitor,
        versions=(version,),
        lifecycle=(lifecycle,),
        schedule_state=state,
        occurrences=(occurrence,),
        leases=(lease,),
        runs=(run,),
        findings=(finding,),
        checkpoints=(checkpoint,),
    )

    assert finding.evidence_id == "evidence-a"
    assert checkpoint.run_id == finding.run_id == state.last_run_id
    assert inspection.versions[-1].content_hash == definition.content_hash

    with pytest.raises(ValueError, match="cursor_hash"):
        MonitorCheckpoint(
            id="checkpoint-b",
            agent_id="agent-a",
            monitor_id=monitor.id,
            version=1,
            run_id=run.id,
            cursor=cursor,
            cursor_hash="sha256:" + ("0" * 64),
            created_at=NOW,
        )
    with pytest.raises(ValueError, match="terminal monitor run"):
        MonitorRun(
            id=run.id,
            agent_id=run.agent_id,
            monitor_id=run.monitor_id,
            occurrence_id=run.occurrence_id,
            trigger_id=run.trigger_id,
            attempt=1,
            fencing_token=1,
            status=MonitorRunStatus.SUCCEEDED,
            started_at=NOW,
        )


def test_monitor_models_and_store_have_no_execution_path_imports() -> None:
    package = Path(__file__).parents[3] / "src" / "daita" / "monitors"
    forbidden = {"adapters", "hosting", "llm", "loop", "operations", "storage"}
    for source_path in (package / "models.py", package / "store.py"):
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        imports = {
            name
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for name in (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
        }
        assert not any(
            segment in forbidden
            for imported in imports
            for segment in imported.split(".")
        )
        assert not any(
            isinstance(node, ast.Attribute) and node.attr == "execute"
            for node in ast.walk(tree)
        )
