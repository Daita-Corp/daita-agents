from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
import sqlite3

import pytest

from daita._json import canonical_json
from daita.capabilities import AccessMode, RiskLevel
from daita.events.models import RuntimeEvent
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.loop.models import LoopBudgets, LoopPhase, LoopState, Turn
from daita.operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from daita.operations.governance import ApprovalRequest
from daita.operations.leases import TaskClaimRequest, TaskLease, TaskLeaseGuard
from daita.operations.models import (
    AgentTrigger,
    Evidence,
    Operation,
    OperationStatus,
    Task,
    TaskDependency,
    TaskExecutionFacts,
    TaskStatus,
    TriggerKind,
)
from daita.operations.store import (
    ExpiredTaskLeaseError,
    InMemoryOperationStore,
    InvalidOperationCheckpointError,
    OperationRevisionConflict,
    StaleTaskFenceError,
    TaskClaimConflictError,
    TaskClaimResult,
    TaskDependenciesNotReadyError,
    TaskNotClaimableError,
    VersionedOperation,
)
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 17, 21, 0, tzinfo=timezone.utc)
MAX_LEASE_DURATION_SECONDS = 30.0
TARGET_TASK_ID = "task-target"
TARGET_CALL_ID = "call-target"
PREREQUISITE_TASK_ID = "task-prerequisite"
PREREQUISITE_CALL_ID = "call-prerequisite"


class ProbeClock:
    def __init__(self, now: datetime) -> None:
        self.now = now
        self.calls = 0

    def __call__(self) -> datetime:
        self.calls += 1
        return self.now

    def set(self, now: datetime) -> None:
        self.now = now


def _content_hash(value: object) -> str:
    encoded = canonical_json(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _facts(
    arguments: dict[str, object],
    *,
    access_mode: AccessMode = AccessMode.READ,
    side_effecting: bool = False,
    idempotent: bool = True,
    replay_safe: bool = True,
    idempotency_key: str | None = None,
) -> TaskExecutionFacts:
    return TaskExecutionFacts(
        capability_fingerprint="sha256:" + ("a" * 64),
        arguments_hash=_content_hash(arguments),
        access_mode=access_mode,
        risk=RiskLevel.HIGH if side_effecting else RiskLevel.LOW,
        side_effecting=side_effecting,
        idempotent=idempotent,
        replay_safe=replay_safe,
        idempotency_key=idempotency_key,
    )


def _safe_read_facts(arguments: dict[str, object]) -> TaskExecutionFacts:
    return _facts(arguments)


def _keyed_side_effect_facts(arguments: dict[str, object]) -> TaskExecutionFacts:
    return _facts(
        arguments,
        access_mode=AccessMode.WRITE,
        side_effecting=True,
        idempotent=True,
        replay_safe=True,
        idempotency_key="operation-1:task-target",
    )


def _unsafe_side_effect_facts(arguments: dict[str, object]) -> TaskExecutionFacts:
    return _facts(
        arguments,
        access_mode=AccessMode.WRITE,
        side_effecting=True,
        idempotent=False,
        replay_safe=False,
    )


def _task(
    *,
    task_id: str,
    call_id: str,
    arguments: dict[str, object],
    execution_facts: TaskExecutionFacts,
    status: TaskStatus,
    cancellation_requested: bool = False,
) -> Task:
    capability_id = "fake.write" if execution_facts.side_effecting else "fake.read"
    evidence_ids = (f"evidence-{task_id}",) if status is TaskStatus.SUCCEEDED else ()
    return Task(
        id=task_id,
        operation_id="operation-1",
        turn_id="turn-1",
        call_id=call_id,
        capability_id=capability_id,
        executor_id="fake.executor",
        status=status,
        attempt=1,
        arguments=arguments,
        execution_facts=execution_facts,
        evidence_ids=evidence_ids,
        error_code="task_failed" if status is TaskStatus.FAILED else None,
        cancellation_requested=cancellation_requested,
        manual_recovery_reason=(
            "operator_review_required"
            if status is TaskStatus.MANUAL_RECOVERY_REQUIRED
            else None
        ),
        created_at=NOW,
        updated_at=NOW,
    )


def _evidence(task: Task) -> Evidence:
    payload = {"value": task.id}
    return Evidence(
        id=f"evidence-{task.id}",
        operation_id=task.operation_id,
        task_id=task.id,
        turn_id=task.turn_id,
        capability_id=task.capability_id,
        executor_id=task.executor_id,
        kind="fake.result",
        schema_version=1,
        attempt=task.attempt,
        accepted=True,
        payload=payload,
        content_hash=_content_hash(payload),
        created_at=NOW,
    )


def _snapshot(
    *,
    target_status: TaskStatus = TaskStatus.READY,
    target_facts: TaskExecutionFacts | None = None,
    cancellation_requested: bool = False,
    prerequisite_status: TaskStatus | None = None,
) -> OperationSnapshot:
    target_arguments = {"key": "target"}
    if target_facts is None:
        target_facts = _safe_read_facts(target_arguments)
    target = _task(
        task_id=TARGET_TASK_ID,
        call_id=TARGET_CALL_ID,
        arguments=target_arguments,
        execution_facts=target_facts,
        status=target_status,
        cancellation_requested=cancellation_requested,
    )

    tasks: list[Task] = []
    dependencies: tuple[TaskDependency, ...] = ()
    if prerequisite_status is not None:
        prerequisite_arguments = {"key": "prerequisite"}
        prerequisite = _task(
            task_id=PREREQUISITE_TASK_ID,
            call_id=PREREQUISITE_CALL_ID,
            arguments=prerequisite_arguments,
            execution_facts=_safe_read_facts(prerequisite_arguments),
            status=prerequisite_status,
        )
        tasks.append(prerequisite)
        dependencies = (
            TaskDependency(
                operation_id="operation-1",
                task_id=TARGET_TASK_ID,
                prerequisite_task_id=PREREQUISITE_TASK_ID,
            ),
        )
    tasks.append(target)

    evidence = tuple(
        _evidence(task) for task in tasks if task.status is TaskStatus.SUCCEEDED
    )
    trigger = AgentTrigger(
        id="trigger-1",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        session_id="session-1",
        payload={"message": "Run the target task."},
        created_at=NOW,
    )
    waiting_for_approval = target_status is TaskStatus.WAITING_FOR_APPROVAL
    operation = Operation(
        id="operation-1",
        agent_id=trigger.agent_id,
        trigger_id=trigger.id,
        session_id=trigger.session_id,
        status=(
            OperationStatus.WAITING_FOR_APPROVAL
            if waiting_for_approval
            else OperationStatus.RUNNING
        ),
        created_at=NOW,
        updated_at=NOW,
    )
    approval_id = "approval-target"
    approvals: tuple[ApprovalRequest, ...] = ()
    if waiting_for_approval:
        approvals = (
            ApprovalRequest(
                id=approval_id,
                operation_id=operation.id,
                task_id=target.id,
                task_fingerprint="sha256:" + ("b" * 64),
                policy_fingerprint="sha256:" + ("c" * 64),
                requested_at=NOW,
            ),
        )
    tool_calls = tuple(
        ToolCall(id=task.call_id, name="fake.action", arguments=task.arguments)
        for task in tasks
    )
    model_request = ModelRequest(
        operation_id=operation.id,
        turn_id="turn-1",
        messages=(
            CanonicalMessage(
                agent_id=operation.agent_id,
                operation_id=operation.id,
                session_id=operation.session_id,
                turn_id="turn-1",
                role=MessageRole.USER,
                content=(TextBlock("Run the target task."),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="fake.action",
                description="Perform a deterministic test action.",
                input_schema={"type": "object"},
            ),
        ),
    )
    model_response = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=tool_calls,
    )
    model_call = ModelCall(
        id="model-call-1",
        operation_id=operation.id,
        turn_id="turn-1",
        provider_id="mock:scripted",
        request=model_request,
        response=model_response,
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )
    turn = Turn(
        id="turn-1",
        operation_id=operation.id,
        number=1,
        model_request_id=model_call.id,
        model_response_id=model_call.id,
        created_at=NOW,
    )
    events = tuple(
        RuntimeEvent(
            id=event_id,
            type=event_type,
            agent_id=operation.agent_id,
            operation_id=operation.id,
            session_id=operation.session_id,
            created_at=NOW,
        )
        for event_id, event_type in (
            ("event-trigger-received", "trigger.received"),
            ("event-operation-created", "operation.created"),
        )
    )
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=(
                LoopPhase.AWAITING_APPROVAL
                if waiting_for_approval
                else LoopPhase.AWAITING_EXECUTION
            ),
            turn_count=1,
            action_count=len(tasks),
            waiting_approval_id=approval_id if waiting_for_approval else None,
        ),
        budgets=LoopBudgets(),
        turns=(turn,),
        model_calls=(model_call,),
        readiness=(),
        tasks=tuple(tasks),
        task_dependencies=dependencies,
        task_leases=(),
        approvals=approvals,
        evidence=evidence,
        observations=(),
        events=events,
    )


def _target(snapshot: OperationSnapshot) -> Task:
    return next(task for task in snapshot.tasks if task.id == TARGET_TASK_ID)


def _task_event(
    snapshot: OperationSnapshot,
    *,
    event_id: str,
    event_type: str,
    created_at: datetime,
    evidence_id: str | None = None,
    payload: dict[str, object] | None = None,
) -> RuntimeEvent:
    task = _target(snapshot)
    return RuntimeEvent(
        id=event_id,
        type=event_type,
        agent_id=snapshot.operation.agent_id,
        operation_id=snapshot.operation.id,
        session_id=snapshot.operation.session_id,
        turn_id=task.turn_id,
        model_call_id="model-call-1",
        call_id=task.call_id,
        task_id=task.id,
        evidence_id=evidence_id,
        capability_id=task.capability_id,
        executor_id=task.executor_id,
        payload={} if payload is None else payload,
        created_at=created_at,
    )


def _claim_request(
    snapshot: OperationSnapshot,
    *,
    holder_id: str = "worker-1",
    duration_seconds: float = 10.0,
    event_id: str = "event-task-claimed-1",
) -> TaskClaimRequest:
    return TaskClaimRequest(
        operation_id=snapshot.operation.id,
        task_id=TARGET_TASK_ID,
        holder_id=holder_id,
        lease_duration_seconds=duration_seconds,
        event=_task_event(
            snapshot,
            event_id=event_id,
            event_type="task.claimed",
            created_at=NOW - timedelta(days=1),
            payload={
                "attempt": 999,
                "fencing_token": 999,
                "holder_id": "untrusted-holder",
            },
        ),
    )


def _store(clock: ProbeClock) -> InMemoryOperationStore:
    return InMemoryOperationStore(
        clock=clock,
        max_lease_duration_seconds=MAX_LEASE_DURATION_SECONDS,
    )


async def _create(
    snapshot: OperationSnapshot,
    clock: ProbeClock,
) -> tuple[InMemoryOperationStore, VersionedOperation]:
    store = _store(clock)
    created = await store.create(snapshot)
    return store, created.operation


def _replace_target(
    snapshot: OperationSnapshot,
    target: Task,
) -> tuple[Task, ...]:
    return tuple(target if task.id == target.id else task for task in snapshot.tasks)


def _replace_lease(
    snapshot: OperationSnapshot,
    current: TaskLease,
    replacement: TaskLease,
) -> tuple[TaskLease, ...]:
    return tuple(
        replacement if lease == current else lease for lease in snapshot.task_leases
    )


def _guard(lease: TaskLease) -> TaskLeaseGuard:
    return TaskLeaseGuard(
        operation_id=lease.operation_id,
        task_id=lease.task_id,
        holder_id=lease.holder_id,
        attempt=lease.attempt,
        fencing_token=lease.fencing_token,
    )


def _start_candidate(claim: TaskClaimResult, at: datetime) -> OperationSnapshot:
    snapshot = claim.commit_result.operation.snapshot
    task = replace(claim.task, status=TaskStatus.RUNNING, updated_at=at)
    lease = replace(claim.lease, started_at=at)
    event = _task_event(
        snapshot,
        event_id=f"event-executor-started-{lease.attempt}",
        event_type="executor.started",
        created_at=at,
        payload={
            "attempt": lease.attempt,
            "fencing_token": lease.fencing_token,
        },
    )
    return replace(
        snapshot,
        operation=replace(snapshot.operation, updated_at=at),
        tasks=_replace_target(snapshot, task),
        task_leases=_replace_lease(snapshot, claim.lease, lease),
        events=(*snapshot.events, event),
    )


def _terminal_candidate(
    started: VersionedOperation,
    *,
    at: datetime,
    outcome: str,
) -> tuple[OperationSnapshot, TaskLeaseGuard]:
    snapshot = started.snapshot
    running = _target(snapshot)
    active = next(
        lease
        for lease in reversed(snapshot.task_leases)
        if lease.task_id == running.id and lease.released_at is None
    )
    guard = _guard(active)
    events: tuple[RuntimeEvent, ...]
    evidence_suffix: tuple[Evidence, ...]
    if outcome == "success":
        evidence = Evidence(
            id="evidence-terminal",
            operation_id=running.operation_id,
            task_id=running.id,
            turn_id=running.turn_id,
            capability_id=running.capability_id,
            executor_id=running.executor_id,
            kind="fake.result",
            schema_version=1,
            attempt=running.attempt,
            accepted=True,
            payload={"value": "accepted"},
            content_hash=_content_hash({"value": "accepted"}),
            created_at=at,
        )
        terminal = replace(
            running,
            status=TaskStatus.SUCCEEDED,
            evidence_ids=(evidence.id,),
            updated_at=at,
        )
        released = replace(
            active,
            released_at=at,
            release_reason="completed",
        )
        events = (
            _task_event(
                snapshot,
                event_id="event-executor-completed",
                event_type="executor.completed",
                created_at=at,
            ),
            _task_event(
                snapshot,
                event_id="event-evidence-accepted",
                event_type="evidence.accepted",
                created_at=at,
                evidence_id=evidence.id,
            ),
            _task_event(
                snapshot,
                event_id="event-task-succeeded",
                event_type="task.succeeded",
                created_at=at,
                evidence_id=evidence.id,
            ),
        )
        evidence_suffix = (evidence,)
    elif outcome == "failure":
        terminal = replace(
            running,
            status=TaskStatus.FAILED,
            error_code="executor_failed",
            updated_at=at,
        )
        released = replace(
            active,
            released_at=at,
            release_reason="executor_failed",
        )
        events = (
            _task_event(
                snapshot,
                event_id="event-task-failed",
                event_type="task.failed",
                created_at=at,
            ),
            _task_event(
                snapshot,
                event_id="event-executor-failed",
                event_type="executor.failed",
                created_at=at,
            ),
        )
        evidence_suffix = ()
    else:
        raise AssertionError(f"unknown terminal outcome: {outcome}")
    return (
        replace(
            snapshot,
            operation=replace(snapshot.operation, updated_at=at),
            tasks=_replace_target(snapshot, terminal),
            task_leases=_replace_lease(snapshot, active, released),
            evidence=(*snapshot.evidence, *evidence_suffix),
            events=(*snapshot.events, *events),
        ),
        guard,
    )


def _renewal_candidate(
    claim: TaskClaimResult,
    *,
    at: datetime,
    duration_seconds: float,
) -> OperationSnapshot:
    snapshot = claim.commit_result.operation.snapshot
    lease = replace(
        claim.lease,
        expires_at=at + timedelta(seconds=duration_seconds),
        renewed_at=at,
    )
    event = _task_event(
        snapshot,
        event_id=f"event-task-lease-renewed-{lease.attempt}",
        event_type="task.lease_renewed",
        created_at=at,
        payload={
            "attempt": lease.attempt,
            "fencing_token": lease.fencing_token,
            "expires_at": lease.expires_at.isoformat(),
        },
    )
    return replace(
        snapshot,
        operation=replace(snapshot.operation, updated_at=at),
        task_leases=_replace_lease(snapshot, claim.lease, lease),
        events=(*snapshot.events, event),
    )


def _recovery_candidate(
    operation: VersionedOperation,
    *,
    at: datetime,
    target_status: TaskStatus,
    release_reason: str,
) -> tuple[OperationSnapshot, TaskLeaseGuard]:
    snapshot = operation.snapshot
    task = _target(snapshot)
    lease = next(
        lease
        for lease in reversed(snapshot.task_leases)
        if lease.task_id == task.id and lease.released_at is None
    )
    recovered_task = replace(
        task,
        status=target_status,
        updated_at=at,
        manual_recovery_reason=(
            "unknown_side_effect_outcome"
            if target_status is TaskStatus.MANUAL_RECOVERY_REQUIRED
            else None
        ),
    )
    released_lease = replace(
        lease,
        released_at=at,
        release_reason=release_reason,
    )
    event = _task_event(
        snapshot,
        event_id=f"event-task-lease-lost-{lease.attempt}",
        event_type="task.lease_lost",
        created_at=at,
        payload={
            "attempt": lease.attempt,
            "fencing_token": lease.fencing_token,
            "from_status": task.status.value,
            "to_status": target_status.value,
            "reason": release_reason,
        },
    )
    return (
        replace(
            snapshot,
            operation=replace(snapshot.operation, updated_at=at),
            tasks=_replace_target(snapshot, recovered_task),
            task_leases=_replace_lease(snapshot, lease, released_lease),
            events=(*snapshot.events, event),
        ),
        _guard(lease),
    )


async def test_claim_uses_one_authoritative_clock_read_and_normalizes_event() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    request = _claim_request(snapshot)

    result = await store.claim_task(request, expected_revision=created.revision)

    assert clock.calls == 1
    assert result.task.status is TaskStatus.CLAIMED
    assert result.task.attempt == 1
    assert result.lease.acquired_at == NOW
    assert result.lease.expires_at == NOW + timedelta(seconds=10)
    assert result.lease.attempt == result.task.attempt
    assert result.lease.fencing_token == 1
    assert result.commit_result.operation.revision == created.revision + 1
    assert len(result.commit_result.committed_events) == 1
    event = result.commit_result.committed_events[0]
    assert event.id == request.event.id
    assert event.created_at == NOW
    assert event.payload != request.event.payload
    assert event.payload["holder_id"] == result.lease.holder_id
    assert event.payload["attempt"] == result.lease.attempt
    assert event.payload["fencing_token"] == result.lease.fencing_token


async def test_claim_requires_all_persisted_dependencies_to_have_succeeded() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot(prerequisite_status=TaskStatus.PENDING)
    store, created = await _create(snapshot, clock)
    before = await store.load(snapshot.operation.id)

    with pytest.raises(TaskDependenciesNotReadyError) as caught:
        await store.claim_task(
            _claim_request(snapshot),
            expected_revision=created.revision,
        )

    assert caught.value.dependency_ids == (PREREQUISITE_TASK_ID,)
    assert await store.load(snapshot.operation.id) == before


async def test_claim_accepts_ready_task_after_all_dependencies_succeed() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot(prerequisite_status=TaskStatus.SUCCEEDED)
    store, created = await _create(snapshot, clock)

    result = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )

    assert result.task.status is TaskStatus.CLAIMED
    assert result.lease.task_id == TARGET_TASK_ID


@pytest.mark.parametrize(
    ("status", "cancellation_requested"),
    (
        (TaskStatus.SUCCEEDED, False),
        (TaskStatus.FAILED, False),
        (TaskStatus.CANCELLED, False),
        (TaskStatus.MANUAL_RECOVERY_REQUIRED, False),
        (TaskStatus.WAITING_FOR_APPROVAL, False),
        (TaskStatus.READY, True),
    ),
)
async def test_terminal_approval_manual_and_cancelled_tasks_are_not_claimable(
    status: TaskStatus,
    cancellation_requested: bool,
) -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot(
        target_status=status,
        cancellation_requested=cancellation_requested,
    )
    store, created = await _create(snapshot, clock)
    before = await store.load(snapshot.operation.id)

    with pytest.raises(TaskNotClaimableError):
        await store.claim_task(
            _claim_request(snapshot),
            expected_revision=created.revision,
        )

    assert await store.load(snapshot.operation.id) == before


async def test_two_concurrent_claimers_have_exactly_one_winner() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    requests = (
        _claim_request(
            snapshot,
            holder_id="worker-1",
            event_id="event-task-claimed-worker-1",
        ),
        _claim_request(
            snapshot,
            holder_id="worker-2",
            event_id="event-task-claimed-worker-2",
        ),
    )

    outcomes = await asyncio.gather(
        *(
            store.claim_task(request, expected_revision=created.revision)
            for request in requests
        ),
        return_exceptions=True,
    )

    winners = [outcome for outcome in outcomes if isinstance(outcome, TaskClaimResult)]
    losers = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
    assert len(winners) == 1
    assert len(losers) == 1
    assert isinstance(
        losers[0],
        (OperationRevisionConflict, TaskClaimConflictError),
    )
    persisted = await store.load(snapshot.operation.id)
    assert persisted.revision == created.revision + 1
    assert len(persisted.snapshot.task_leases) == 1
    assert persisted.snapshot.task_leases[0].holder_id == winners[0].lease.holder_id


async def test_claim_duration_cannot_exceed_the_adapter_maximum() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    before = await store.load(snapshot.operation.id)

    with pytest.raises(
        (ValueError, InvalidOperationCheckpointError),
        match="lease.*duration|duration.*lease|maximum",
    ):
        await store.claim_task(
            _claim_request(
                snapshot,
                duration_seconds=MAX_LEASE_DURATION_SECONDS + 0.001,
            ),
            expected_revision=created.revision,
        )

    assert await store.load(snapshot.operation.id) == before


async def test_live_lease_can_be_renewed_only_by_strictly_extending_expiry() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(NOW + timedelta(seconds=5))
    candidate = _renewal_candidate(claim, at=clock.now, duration_seconds=10)

    renewed = await store.renew_task_lease(
        candidate,
        expected_revision=claim.commit_result.operation.revision,
        guard=_guard(claim.lease),
        lease_duration_seconds=10,
    )

    renewed_lease = renewed.operation.snapshot.task_leases[-1]
    assert renewed_lease.renewed_at == clock.now
    assert renewed_lease.expires_at == clock.now + timedelta(seconds=10)
    assert renewed_lease.expires_at > claim.lease.expires_at


async def test_renewal_at_expiry_is_expired_and_has_zero_delta() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(claim.lease.expires_at)
    candidate = _renewal_candidate(claim, at=clock.now, duration_seconds=10)
    before = await store.load(snapshot.operation.id)

    with pytest.raises(ExpiredTaskLeaseError):
        await store.renew_task_lease(
            candidate,
            expected_revision=before.revision,
            guard=_guard(claim.lease),
            lease_duration_seconds=10,
        )

    assert await store.load(snapshot.operation.id) == before


async def test_fenced_claimed_to_running_transition_commits_atomically() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(NOW + timedelta(seconds=1))
    candidate = _start_candidate(claim, clock.now)

    started = await store.commit_fenced(
        candidate,
        expected_revision=claim.commit_result.operation.revision,
        guard=_guard(claim.lease),
    )

    assert _target(started.operation.snapshot).status is TaskStatus.RUNNING
    assert started.operation.snapshot.task_leases[-1].started_at == clock.now
    assert started.committed_events[-1].type == "executor.started"


async def test_stale_fence_rejection_has_zero_checkpoint_delta() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(NOW + timedelta(seconds=1))
    candidate = _start_candidate(claim, clock.now)
    stale_guard = replace(
        _guard(claim.lease),
        fencing_token=claim.lease.fencing_token + 1,
    )
    before = await store.load(snapshot.operation.id)

    with pytest.raises(StaleTaskFenceError):
        await store.commit_fenced(
            candidate,
            expected_revision=before.revision,
            guard=stale_guard,
        )

    assert await store.load(snapshot.operation.id) == before


async def test_ordinary_commit_cannot_forge_claimed_state_without_a_lease() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    at = NOW + timedelta(seconds=1)
    target = replace(_target(snapshot), status=TaskStatus.CLAIMED, updated_at=at)
    forged = replace(
        snapshot,
        operation=replace(snapshot.operation, updated_at=at),
        tasks=_replace_target(snapshot, target),
        events=(
            *snapshot.events,
            _task_event(
                snapshot,
                event_id="event-forged-claim",
                event_type="task.claimed",
                created_at=at,
            ),
        ),
    )

    with pytest.raises(InvalidOperationCheckpointError, match="claim|lease|fenc"):
        await store.commit(forged, expected_revision=created.revision)

    assert await store.load(snapshot.operation.id) == created


async def test_fenced_start_cannot_erase_persisted_cancellation_intent() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(NOW + timedelta(seconds=1))
    claimed = claim.commit_result.operation.snapshot
    cancellation_requested = replace(
        _target(claimed),
        cancellation_requested=True,
        updated_at=clock.now,
    )
    cancelled = await store.commit(
        replace(
            claimed,
            operation=replace(claimed.operation, updated_at=clock.now),
            tasks=_replace_target(claimed, cancellation_requested),
            events=(
                *claimed.events,
                _task_event(
                    claimed,
                    event_id="event-task-cancellation-requested",
                    event_type="task.cancellation_requested",
                    created_at=clock.now,
                ),
            ),
        ),
        expected_revision=claim.commit_result.operation.revision,
    )
    clock.set(NOW + timedelta(seconds=2))
    current = cancelled.operation.snapshot
    active = current.task_leases[-1]
    forged_task = replace(
        _target(current),
        status=TaskStatus.RUNNING,
        cancellation_requested=False,
        updated_at=clock.now,
    )
    forged_lease = replace(active, started_at=clock.now)
    forged = replace(
        current,
        operation=replace(current.operation, updated_at=clock.now),
        tasks=_replace_target(current, forged_task),
        task_leases=_replace_lease(current, active, forged_lease),
        events=(
            *current.events,
            _task_event(
                current,
                event_id="event-forged-start-after-cancellation",
                event_type="executor.started",
                created_at=clock.now,
            ),
        ),
    )

    with pytest.raises(InvalidOperationCheckpointError, match="cancellation"):
        await store.commit_fenced(
            forged,
            expected_revision=cancelled.operation.revision,
            guard=_guard(active),
        )

    assert await store.load(snapshot.operation.id) == cancelled.operation


async def test_fenced_start_cannot_mutate_loop_progression() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(NOW + timedelta(seconds=1))
    candidate = replace(
        _start_candidate(claim, clock.now),
        loop_state=replace(
            claim.commit_result.operation.snapshot.loop_state,
            turn_count=2,
        ),
    )

    with pytest.raises(InvalidOperationCheckpointError, match="loop state|progress"):
        await store.commit_fenced(
            candidate,
            expected_revision=claim.commit_result.operation.revision,
            guard=_guard(claim.lease),
        )


async def test_fenced_commit_cannot_reorder_task_history() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot(prerequisite_status=TaskStatus.SUCCEEDED)
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(NOW + timedelta(seconds=1))
    candidate = _start_candidate(claim, clock.now)
    candidate = replace(candidate, tasks=tuple(reversed(candidate.tasks)))

    with pytest.raises(InvalidOperationCheckpointError, match="order|identit"):
        await store.commit_fenced(
            candidate,
            expected_revision=claim.commit_result.operation.revision,
            guard=_guard(claim.lease),
        )


async def test_fenced_terminal_state_requires_its_canonical_outcome_event() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(NOW + timedelta(seconds=1))
    started = await store.commit_fenced(
        _start_candidate(claim, clock.now),
        expected_revision=claim.commit_result.operation.revision,
        guard=_guard(claim.lease),
    )
    clock.set(NOW + timedelta(seconds=2))
    current = started.operation.snapshot
    running = _target(current)
    active = current.task_leases[-1]
    failed = replace(
        running,
        status=TaskStatus.FAILED,
        error_code="executor_failed",
        updated_at=clock.now,
    )
    released = replace(
        active,
        released_at=clock.now,
        release_reason="executor_failed",
    )
    forged = replace(
        current,
        operation=replace(current.operation, updated_at=clock.now),
        tasks=_replace_target(current, failed),
        task_leases=_replace_lease(current, active, released),
        events=(
            *current.events,
            _task_event(
                current,
                event_id="event-unrelated-terminal",
                event_type="checkpoint.updated",
                created_at=clock.now,
            ),
        ),
    )

    with pytest.raises(InvalidOperationCheckpointError, match="task.failed"):
        await store.commit_fenced(
            forged,
            expected_revision=started.operation.revision,
            guard=_guard(active),
        )


async def test_expired_unsafe_running_cancellation_fails_closed_to_manual() -> None:
    arguments = {"key": "target"}
    clock = ProbeClock(NOW)
    snapshot = _snapshot(target_facts=_unsafe_side_effect_facts(arguments))
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    clock.set(NOW + timedelta(seconds=1))
    started = await store.commit_fenced(
        _start_candidate(claim, clock.now),
        expected_revision=claim.commit_result.operation.revision,
        guard=_guard(claim.lease),
    )
    current = started.operation.snapshot
    running = replace(
        _target(current),
        cancellation_requested=True,
        updated_at=clock.now,
    )
    cancellation = await store.commit(
        replace(
            current,
            tasks=_replace_target(current, running),
            events=(
                *current.events,
                _task_event(
                    current,
                    event_id="event-running-cancellation-requested",
                    event_type="task.cancellation_requested",
                    created_at=clock.now,
                ),
            ),
        ),
        expected_revision=started.operation.revision,
    )
    clock.set(claim.lease.expires_at)
    candidate, guard = _recovery_candidate(
        cancellation.operation,
        at=clock.now,
        target_status=TaskStatus.MANUAL_RECOVERY_REQUIRED,
        release_reason="expired_unknown_outcome",
    )

    recovered = await store.recover_expired_task(
        candidate,
        expected_revision=cancellation.operation.revision,
        guard=guard,
    )

    assert (
        _target(recovered.operation.snapshot).status
        is TaskStatus.MANUAL_RECOVERY_REQUIRED
    )
    assert (
        recovered.operation.snapshot.task_leases[-1].release_reason
        == "expired_unknown_outcome"
    )


@pytest.mark.parametrize(
    ("case", "facts_factory", "started", "expected_status", "release_reason"),
    (
        (
            "never_started",
            _unsafe_side_effect_facts,
            False,
            TaskStatus.READY,
            "expired_before_start",
        ),
        (
            "safe_read",
            _safe_read_facts,
            True,
            TaskStatus.READY,
            "expired_replay_safe",
        ),
        (
            "keyed_side_effect",
            _keyed_side_effect_facts,
            True,
            TaskStatus.READY,
            "expired_replay_safe",
        ),
        (
            "unsafe_side_effect",
            _unsafe_side_effect_facts,
            True,
            TaskStatus.MANUAL_RECOVERY_REQUIRED,
            "expired_unknown_outcome",
        ),
    ),
    ids=lambda value: value if isinstance(value, str) else None,
)
async def test_expired_task_recovery_uses_the_explicit_safety_matrix(
    case: str,
    facts_factory: object,
    started: bool,
    expected_status: TaskStatus,
    release_reason: str,
) -> None:
    del case
    arguments = {"key": "target"}
    assert callable(facts_factory)
    facts = facts_factory(arguments)
    clock = ProbeClock(NOW)
    snapshot = _snapshot(target_facts=facts)
    store, created = await _create(snapshot, clock)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.revision,
    )
    current = claim.commit_result.operation
    if started:
        clock.set(NOW + timedelta(seconds=1))
        start = await store.commit_fenced(
            _start_candidate(claim, clock.now),
            expected_revision=current.revision,
            guard=_guard(claim.lease),
        )
        current = start.operation

    clock.set(claim.lease.expires_at)
    candidate, guard = _recovery_candidate(
        current,
        at=clock.now,
        target_status=expected_status,
        release_reason=release_reason,
    )
    recovered = await store.recover_expired_task(
        candidate,
        expected_revision=current.revision,
        guard=guard,
    )

    task = _target(recovered.operation.snapshot)
    lease = recovered.operation.snapshot.task_leases[-1]
    assert task.status is expected_status
    assert task.execution_facts == facts
    assert lease.released_at == clock.now
    assert lease.release_reason == release_reason
    if expected_status is TaskStatus.MANUAL_RECOVERY_REQUIRED:
        assert task.manual_recovery_reason == "unknown_side_effect_outcome"
    else:
        assert task.manual_recovery_reason is None


async def test_reclaim_after_expiry_increments_attempt_and_fencing_token() -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store, created = await _create(snapshot, clock)
    first = await store.claim_task(
        _claim_request(snapshot, event_id="event-task-claimed-attempt-1"),
        expected_revision=created.revision,
    )
    clock.set(first.lease.expires_at)
    candidate, guard = _recovery_candidate(
        first.commit_result.operation,
        at=clock.now,
        target_status=TaskStatus.READY,
        release_reason="expired_before_start",
    )
    recovered = await store.recover_expired_task(
        candidate,
        expected_revision=first.commit_result.operation.revision,
        guard=guard,
    )
    clock.set(clock.now + timedelta(microseconds=1))

    second = await store.claim_task(
        _claim_request(
            recovered.operation.snapshot,
            event_id="event-task-claimed-attempt-2",
        ),
        expected_revision=recovered.operation.revision,
    )

    assert second.task.attempt == first.task.attempt + 1
    assert second.lease.attempt == first.lease.attempt + 1
    assert second.lease.fencing_token == first.lease.fencing_token + 1
    assert len(second.commit_result.operation.snapshot.task_leases) == 2
    assert (
        second.commit_result.operation.snapshot.task_leases[0].released_at is not None
    )
    assert second.commit_result.operation.snapshot.task_leases[1] == second.lease

    before_stale_attempt = await store.load(snapshot.operation.id)
    with pytest.raises(StaleTaskFenceError):
        await store.commit_fenced(
            _start_candidate(first, NOW + timedelta(seconds=1)),
            expected_revision=before_stale_attempt.revision,
            guard=_guard(first.lease),
        )
    assert await store.load(snapshot.operation.id) == before_stale_attempt


async def test_sqlite_claim_round_trips_authoritative_state_after_reopen(
    tmp_path: Path,
) -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    path = tmp_path / "claim.db"
    store = await SQLiteOperationStore.open(
        path,
        clock=clock,
        max_lease_duration_seconds=MAX_LEASE_DURATION_SECONDS,
    )
    created = await store.create(snapshot)
    claim = await store.claim_task(
        _claim_request(snapshot),
        expected_revision=created.operation.revision,
    )
    await store.close()

    reopened = await SQLiteOperationStore.open(path, clock=clock)
    try:
        assert (
            await reopened.load(snapshot.operation.id) == claim.commit_result.operation
        )
        assert claim.lease.acquired_at == NOW
        assert claim.commit_result.committed_events[0].created_at == NOW
    finally:
        await reopened.close()


async def test_two_sqlite_connections_have_one_claim_winner(tmp_path: Path) -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    path = tmp_path / "claim-race.db"
    first = await SQLiteOperationStore.open(path, clock=clock)
    second = await SQLiteOperationStore.open(path, clock=clock)
    try:
        created = await first.create(snapshot)
        requests = (
            _claim_request(
                snapshot,
                holder_id="worker-1",
                event_id="event-sqlite-claim-1",
            ),
            _claim_request(
                snapshot,
                holder_id="worker-2",
                event_id="event-sqlite-claim-2",
            ),
        )
        outcomes = await asyncio.gather(
            *(
                store.claim_task(request, expected_revision=created.operation.revision)
                for store, request in zip((first, second), requests, strict=True)
            ),
            return_exceptions=True,
        )

        assert sum(isinstance(item, TaskClaimResult) for item in outcomes) == 1
        assert (
            sum(
                isinstance(item, (OperationRevisionConflict, TaskClaimConflictError))
                for item in outcomes
            )
            == 1
        )
        persisted = await first.load(snapshot.operation.id)
        assert persisted.revision == created.operation.revision + 1
        assert len(persisted.snapshot.task_leases) == 1
    finally:
        await first.close()
        await second.close()


async def test_sqlite_renewal_and_expiry_rejection_are_atomic(
    tmp_path: Path,
) -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store = await SQLiteOperationStore.open(tmp_path / "renew.db", clock=clock)
    try:
        created = await store.create(snapshot)
        claim = await store.claim_task(
            _claim_request(snapshot),
            expected_revision=created.operation.revision,
        )
        clock.set(NOW + timedelta(seconds=5))
        renewed = await store.renew_task_lease(
            _renewal_candidate(claim, at=clock.now, duration_seconds=10),
            expected_revision=claim.commit_result.operation.revision,
            guard=_guard(claim.lease),
            lease_duration_seconds=10,
        )
        active = renewed.operation.snapshot.task_leases[-1]
        assert active.expires_at == clock.now + timedelta(seconds=10)

        clock.set(active.expires_at)
        before = await store.load(snapshot.operation.id)
        with pytest.raises(ExpiredTaskLeaseError):
            await store.renew_task_lease(
                replace(
                    renewed.operation.snapshot,
                    operation=replace(
                        renewed.operation.snapshot.operation,
                        updated_at=clock.now,
                    ),
                    task_leases=_replace_lease(
                        renewed.operation.snapshot,
                        active,
                        replace(
                            active,
                            renewed_at=clock.now,
                            expires_at=clock.now + timedelta(seconds=10),
                        ),
                    ),
                    events=(
                        *renewed.operation.snapshot.events,
                        _task_event(
                            renewed.operation.snapshot,
                            event_id="event-expired-sqlite-renewal",
                            event_type="task.lease_renewed",
                            created_at=clock.now,
                        ),
                    ),
                ),
                expected_revision=before.revision,
                guard=_guard(active),
                lease_duration_seconds=10,
            )
        assert await store.load(snapshot.operation.id) == before
    finally:
        await store.close()


async def test_sqlite_expired_recovery_reclaims_with_monotonic_fence(
    tmp_path: Path,
) -> None:
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    path = tmp_path / "reclaim.db"
    store = await SQLiteOperationStore.open(path, clock=clock)
    created = await store.create(snapshot)
    first = await store.claim_task(
        _claim_request(snapshot, event_id="event-sqlite-attempt-1"),
        expected_revision=created.operation.revision,
    )
    clock.set(first.lease.expires_at)
    candidate, guard = _recovery_candidate(
        first.commit_result.operation,
        at=clock.now,
        target_status=TaskStatus.READY,
        release_reason="expired_before_start",
    )
    recovered = await store.recover_expired_task(
        candidate,
        expected_revision=first.commit_result.operation.revision,
        guard=guard,
    )
    clock.set(clock.now + timedelta(microseconds=1))
    second = await store.claim_task(
        _claim_request(
            recovered.operation.snapshot,
            event_id="event-sqlite-attempt-2",
        ),
        expected_revision=recovered.operation.revision,
    )
    before_stale_attempt = await store.load(snapshot.operation.id)
    with pytest.raises(StaleTaskFenceError):
        await store.commit_fenced(
            _start_candidate(first, NOW + timedelta(seconds=1)),
            expected_revision=before_stale_attempt.revision,
            guard=_guard(first.lease),
        )
    assert await store.load(snapshot.operation.id) == before_stale_attempt
    await store.close()

    reopened = await SQLiteOperationStore.open(path, clock=clock)
    try:
        assert (
            await reopened.load(snapshot.operation.id) == second.commit_result.operation
        )
        assert second.task.attempt == 2
        assert second.lease.fencing_token == 2
    finally:
        await reopened.close()


async def test_sqlite_unsafe_started_expiry_persists_manual_recovery(
    tmp_path: Path,
) -> None:
    arguments = {"key": "target"}
    clock = ProbeClock(NOW)
    snapshot = _snapshot(target_facts=_unsafe_side_effect_facts(arguments))
    store = await SQLiteOperationStore.open(tmp_path / "manual.db", clock=clock)
    try:
        created = await store.create(snapshot)
        claim = await store.claim_task(
            _claim_request(snapshot),
            expected_revision=created.operation.revision,
        )
        clock.set(NOW + timedelta(seconds=1))
        started = await store.commit_fenced(
            _start_candidate(claim, clock.now),
            expected_revision=claim.commit_result.operation.revision,
            guard=_guard(claim.lease),
        )
        clock.set(claim.lease.expires_at)
        candidate, guard = _recovery_candidate(
            started.operation,
            at=clock.now,
            target_status=TaskStatus.MANUAL_RECOVERY_REQUIRED,
            release_reason="expired_unknown_outcome",
        )
        recovered = await store.recover_expired_task(
            candidate,
            expected_revision=started.operation.revision,
            guard=guard,
        )

        persisted = await store.load(snapshot.operation.id)
        assert persisted == recovered.operation
        assert _target(persisted.snapshot).status is TaskStatus.MANUAL_RECOVERY_REQUIRED
        assert (
            _target(persisted.snapshot).manual_recovery_reason
            == "unknown_side_effect_outcome"
        )
    finally:
        await store.close()


async def test_sqlite_samples_its_clock_inside_the_write_transaction(
    tmp_path: Path,
) -> None:
    path = tmp_path / "clock-boundary.db"

    class TransactionProbeClock:
        def __init__(self) -> None:
            self.calls = 0
            self.observed_write_lock = False

        def __call__(self) -> datetime:
            self.calls += 1
            contender = sqlite3.connect(path, timeout=0, isolation_level=None)
            try:
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    contender.execute("BEGIN IMMEDIATE")
                self.observed_write_lock = True
            finally:
                contender.close()
            return NOW

    clock = TransactionProbeClock()
    snapshot = _snapshot()
    store = await SQLiteOperationStore.open(path, clock=clock)
    try:
        created = await store.create(snapshot)
        await store.claim_task(
            _claim_request(snapshot),
            expected_revision=created.operation.revision,
        )

        assert clock.calls == 1
        assert clock.observed_write_lock is True
    finally:
        await store.close()


async def test_sqlite_claim_event_failure_rolls_back_the_whole_transition(
    tmp_path: Path,
) -> None:
    path = tmp_path / "claim-rollback.db"
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store = await SQLiteOperationStore.open(path, clock=clock)
    try:
        created = await store.create(snapshot)
        trigger_connection = sqlite3.connect(path)
        try:
            trigger_connection.execute("""
                CREATE TRIGGER abort_task_claim_event
                BEFORE INSERT ON runtime_events
                WHEN NEW.type = 'task.claimed'
                BEGIN
                    SELECT RAISE(ABORT, 'forced claim event failure');
                END
                """)
            trigger_connection.commit()
        finally:
            trigger_connection.close()

        with pytest.raises(sqlite3.IntegrityError, match="forced claim event"):
            await store.claim_task(
                _claim_request(snapshot),
                expected_revision=created.operation.revision,
            )

        assert await store.load(snapshot.operation.id) == created.operation
        inspection = sqlite3.connect(path)
        try:
            assert inspection.execute(
                "SELECT revision FROM operations WHERE id = ?",
                (snapshot.operation.id,),
            ).fetchone() == (1,)
            assert inspection.execute(
                "SELECT status, attempt FROM tasks WHERE operation_id = ? "
                "AND id = ?",
                (snapshot.operation.id, TARGET_TASK_ID),
            ).fetchone() == (TaskStatus.READY.value, 1)
            assert inspection.execute(
                "SELECT COUNT(*) FROM task_leases WHERE operation_id = ?",
                (snapshot.operation.id,),
            ).fetchone() == (0,)
        finally:
            inspection.close()
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("outcome", "abort_event_type"),
    (
        ("success", "evidence.accepted"),
        ("failure", "executor.failed"),
    ),
)
async def test_sqlite_terminal_event_failure_rolls_back_task_lease_and_evidence(
    tmp_path: Path,
    outcome: str,
    abort_event_type: str,
) -> None:
    path = tmp_path / f"terminal-{outcome}-rollback.db"
    clock = ProbeClock(NOW)
    snapshot = _snapshot()
    store = await SQLiteOperationStore.open(path, clock=clock)
    try:
        created = await store.create(snapshot)
        claim = await store.claim_task(
            _claim_request(snapshot),
            expected_revision=created.operation.revision,
        )
        clock.set(NOW + timedelta(seconds=1))
        started = await store.commit_fenced(
            _start_candidate(claim, clock.now),
            expected_revision=claim.commit_result.operation.revision,
            guard=_guard(claim.lease),
        )
        before = started.operation
        candidate, guard = _terminal_candidate(
            before,
            at=NOW + timedelta(seconds=2),
            outcome=outcome,
        )
        trigger_connection = sqlite3.connect(path)
        try:
            trigger_connection.execute(f"""
                CREATE TRIGGER abort_terminal_event
                BEFORE INSERT ON runtime_events
                WHEN NEW.type = '{abort_event_type}'
                BEGIN
                    SELECT RAISE(ABORT, 'forced terminal event failure');
                END
                """)
            trigger_connection.commit()
        finally:
            trigger_connection.close()
        clock.set(NOW + timedelta(seconds=2))

        with pytest.raises(sqlite3.IntegrityError, match="forced terminal event"):
            await store.commit_fenced(
                candidate,
                expected_revision=before.revision,
                guard=guard,
            )

        assert await store.load(snapshot.operation.id) == before
        inspection = sqlite3.connect(path)
        try:
            assert inspection.execute(
                "SELECT revision FROM operations WHERE id = ?",
                (snapshot.operation.id,),
            ).fetchone() == (before.revision,)
            assert inspection.execute(
                "SELECT status FROM tasks WHERE operation_id = ? AND id = ?",
                (snapshot.operation.id, TARGET_TASK_ID),
            ).fetchone() == (TaskStatus.RUNNING.value,)
            assert inspection.execute(
                "SELECT released_at, release_reason FROM task_leases "
                "WHERE operation_id = ? AND task_id = ?",
                (snapshot.operation.id, TARGET_TASK_ID),
            ).fetchone() == (None, None)
            assert inspection.execute(
                "SELECT COUNT(*) FROM evidence WHERE operation_id = ?",
                (snapshot.operation.id,),
            ).fetchone() == (len(before.snapshot.evidence),)
        finally:
            inspection.close()
    finally:
        await store.close()
