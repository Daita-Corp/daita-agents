from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityExecutionError,
    CapabilityRegistry,
    EvidenceCandidate,
    EvidenceValidationError,
    ExecutionRequest,
    Executor,
    RiskLevel,
    ToolView,
)
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
from daita.loop.models import LoopBudgets, LoopPhase, Readiness
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Observation,
    TaskStatus,
    TriggerKind,
)
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.leases import TaskClaimRequest, TaskLeaseGuard
from daita.operations.runtime import (
    OperationRuntime,
    OperationStateError,
    OperationWallTimeExceeded,
    TaskExecutionTimeout,
)
from daita.operations.store import (
    CommitResult,
    InMemoryOperationStore,
    InvalidOperationCheckpointError,
    StaleTaskFenceError,
    TaskClaimResult,
    TaskExecutionStore,
    VersionedOperation,
)
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)


class CandidateExecutor:
    def __init__(self, executor_id: str, candidate: EvidenceCandidate) -> None:
        self.executor_id = executor_id
        self.candidate = candidate
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        return self.candidate


class HangingExecutor(CandidateExecutor):
    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        await asyncio.Event().wait()
        raise AssertionError("runtime timeout must stop the hanging executor")


class SlowCancellationSuppressingExecutor(CandidateExecutor):
    def __init__(
        self,
        executor_id: str,
        candidate: EvidenceCandidate,
        store: TaskExecutionStore,
        *,
        suppression_seconds: float,
    ) -> None:
        super().__init__(executor_id, candidate)
        self._store = store
        self._suppression_seconds = suppression_seconds
        self.at_entry: VersionedOperation | None = None

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        self.at_entry = await self._store.load(request.operation_id)
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            await asyncio.sleep(self._suppression_seconds)
            return self.candidate
        raise AssertionError("executor wait unexpectedly completed")


class IdentityMutatingExecutor(CandidateExecutor):
    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        self.executor_id = "mutated.executor"
        return self.candidate


class InspectingExecutor(CandidateExecutor):
    def __init__(
        self,
        executor_id: str,
        candidate: EvidenceCandidate,
        store: TaskExecutionStore,
    ) -> None:
        super().__init__(executor_id, candidate)
        self._store = store
        self.snapshots_at_entry: list[OperationSnapshot] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        committed = await self._store.load(request.operation_id)
        self.snapshots_at_entry.append(committed.snapshot)
        return self.candidate


class RejectingFencedStartStore(InMemoryOperationStore):
    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        raise InvalidOperationCheckpointError(
            snapshot.operation.id,
            "injected fenced start failure",
        )


class RejectingTerminalFenceStore(InMemoryOperationStore):
    def __init__(self) -> None:
        super().__init__(clock=lambda: NOW)
        self.fenced_calls = 0
        self.before_rejection: VersionedOperation | None = None

    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        self.fenced_calls += 1
        if self.fenced_calls == 2:
            self.before_rejection = await self.load(guard.operation_id)
            raise StaleTaskFenceError(
                guard.operation_id,
                guard.task_id,
                guard.fencing_token,
                guard.fencing_token + 1,
            )
        return await super().commit_fenced(
            snapshot,
            expected_revision=expected_revision,
            guard=guard,
        )


class DelayedFencedStartAcknowledgementStore(InMemoryOperationStore):
    def __init__(self, delay_seconds: float) -> None:
        super().__init__()
        self._delay_seconds = delay_seconds
        self._fenced_calls = 0

    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        committed = await super().commit_fenced(
            snapshot,
            expected_revision=expected_revision,
            guard=guard,
        )
        self._fenced_calls += 1
        if self._fenced_calls == 1:
            await asyncio.sleep(self._delay_seconds)
        return committed


class DriftAfterReadyStore(InMemoryOperationStore):
    def __init__(self, drift: Callable[[], None]) -> None:
        super().__init__(clock=lambda: NOW)
        self._drift = drift

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        committed = await super().commit(
            snapshot,
            expected_revision=expected_revision,
        )
        if committed.committed_events[-1].type == "task.ready":
            self._drift()
        return committed


class MutableClock:
    def __init__(self) -> None:
        self.current = NOW

    def __call__(self) -> datetime:
        return self.current

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)


class AdvanceAfterClaimStore(InMemoryOperationStore):
    def __init__(self, clock: MutableClock) -> None:
        super().__init__(clock=clock)
        self._mutable_clock = clock

    async def claim_task(
        self,
        request: TaskClaimRequest,
        *,
        expected_revision: int,
    ) -> TaskClaimResult:
        result = await super().claim_task(
            request,
            expected_revision=expected_revision,
        )
        self._mutable_clock.advance(10.0)
        return result


def _capability(
    *,
    capability_id: str = "fake.read",
    executor_id: str = "fake.read.executor",
    evidence_kind: str = "fake.read.result",
) -> Capability:
    return Capability(
        id=capability_id,
        owner="loop-lab",
        description=f"Execute {capability_id}.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind=evidence_kind,
        output_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        output_schema_version=1,
        executor_id=executor_id,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


def _registry(
    first_executor: CandidateExecutor,
    *,
    include_other: bool = False,
) -> CapabilityRegistry:
    capabilities = [_capability()]
    executors = [first_executor]
    tool_views = [
        ToolView(
            name="read_fake",
            capability_id="fake.read",
            description="Read one fake value.",
        )
    ]
    if include_other:
        capabilities.append(
            _capability(
                capability_id="fake.other",
                executor_id="fake.other.executor",
                evidence_kind="fake.other.result",
            )
        )
        executors.append(
            CandidateExecutor(
                "fake.other.executor",
                EvidenceCandidate(
                    kind="fake.other.result",
                    schema_version=1,
                    payload={"key": "alpha", "value": "OTHER"},
                ),
            )
        )
        tool_views.append(
            ToolView(
                name="read_other",
                capability_id="fake.other",
                description="Read a different fake value.",
            )
        )
    return CapabilityRegistry(
        capabilities=tuple(capabilities),
        executors=tuple(executors),
        tool_views=tuple(tool_views),
    )


async def _runtime_with_committed_tool_call(
    candidate: EvidenceCandidate,
    *,
    include_other: bool = False,
    id_factory: Callable[[str], str] | None = None,
    executor_override: CandidateExecutor | None = None,
    request_tools: tuple[ToolDefinition, ...] | None = None,
    tool_calls: tuple[ToolCall, ...] | None = None,
    store: TaskExecutionStore | None = None,
    registry_override: CapabilityRegistry | None = None,
    clock: Callable[[], datetime] = lambda: NOW,
    budgets: LoopBudgets | None = None,
    lease_duration_seconds: float = 60.0,
) -> tuple[OperationRuntime, CandidateExecutor, str, str]:
    executor = executor_override or CandidateExecutor("fake.read.executor", candidate)
    registry = registry_override or _registry(executor, include_other=include_other)
    if id_factory is None:
        runtime = OperationRuntime(
            capabilities=registry,
            clock=clock,
            store=store,
            lease_duration_seconds=lease_duration_seconds,
        )
    else:
        runtime = OperationRuntime(
            capabilities=registry,
            clock=clock,
            id_factory=id_factory,
            store=store,
            lease_duration_seconds=lease_duration_seconds,
        )
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-1",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "read alpha"},
            created_at=NOW,
        ),
        budgets=budgets or LoopBudgets(),
    )
    turn = await runtime.begin_turn(started.operation.id)
    request = ModelRequest(
        operation_id=started.operation.id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=started.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Read alpha."),),
            ),
        ),
        tools=(registry.tool_definitions() if request_tools is None else request_tools),
    )
    model_call = await runtime.begin_model_call(
        started.operation.id,
        turn.id,
        "mock:scripted",
        request,
    )
    await runtime.record_model_response(
        started.operation.id,
        model_call.id,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=tool_calls
            or (
                ToolCall(
                    id="call-1",
                    name="read_fake",
                    arguments={"key": "alpha"},
                ),
            ),
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    return runtime, executor, started.operation.id, turn.id


def _proposal(
    operation_id: str,
    turn_id: str,
    *,
    call_id: str = "call-1",
    capability_id: str = "fake.read",
    arguments: Mapping[str, object] | None = None,
) -> ActionProposal:
    return ActionProposal(
        operation_id=operation_id,
        turn_id=turn_id,
        call_id=call_id,
        capability_id=capability_id,
        arguments={"key": "alpha"} if arguments is None else arguments,
        proposed_at=NOW,
    )


async def test_submit_executes_only_after_fenced_start_and_forwards_identity() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    store = InMemoryOperationStore(clock=lambda: NOW)
    executor = InspectingExecutor("fake.read.executor", candidate, store)
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
        store=store,
    )

    evidence = await runtime.submit(_proposal(operation_id, turn_id))

    assert len(executor.snapshots_at_entry) == 1
    at_entry = executor.snapshots_at_entry[0]
    task_at_entry = at_entry.tasks[0]
    assert task_at_entry.status is TaskStatus.RUNNING
    assert len(at_entry.task_leases) == 1
    lease_at_entry = at_entry.task_leases[0]
    assert lease_at_entry.started_at is not None
    assert lease_at_entry.released_at is None

    request = executor.requests[0]
    assert request.executor_id == task_at_entry.executor_id
    assert request.attempt == lease_at_entry.attempt == task_at_entry.attempt
    assert request.fencing_token == lease_at_entry.fencing_token
    assert request.idempotency_key == task_at_entry.execution_facts.idempotency_key

    final = await runtime.inspect(operation_id)
    assert final.tasks[0].status is TaskStatus.SUCCEEDED
    assert final.tasks[0].evidence_ids == (evidence.id,)
    assert final.task_leases[0].released_at is not None
    assert final.task_leases[0].release_reason == "completed"
    assert [
        event.type for event in final.events if event.task_id == final.tasks[0].id
    ] == [
        "task.created",
        "task.ready",
        "task.claimed",
        "executor.started",
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
    ]


async def test_wall_deadline_crossed_after_claim_blocks_executor_io() -> None:
    clock = MutableClock()
    store = AdvanceAfterClaimStore(clock)
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        store=store,
        clock=clock,
        budgets=LoopBudgets(
            max_wall_time_seconds=5.0,
            task_timeout_seconds=1.0,
        ),
    )

    with pytest.raises(OperationWallTimeExceeded):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    task = snapshot.tasks[0]
    lease = snapshot.task_leases[0]
    assert executor.requests == []
    assert task.status is TaskStatus.FAILED
    assert task.error_code == "task_timeout"
    assert task.evidence_ids == ()
    assert lease.started_at is not None
    assert lease.released_at is not None
    assert lease.release_reason == "task_timeout"
    assert snapshot.evidence == ()
    assert [event.type for event in snapshot.events if event.task_id == task.id] == [
        "task.created",
        "task.ready",
        "task.claimed",
        "executor.started",
        "task.failed",
        "executor.failed",
    ]


async def test_executor_timeout_uses_store_interval_despite_clock_skew() -> None:
    store_now = NOW + timedelta(seconds=100)
    runtime_now = NOW
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    executor = HangingExecutor("fake.read.executor", candidate)
    store = InMemoryOperationStore(clock=lambda: store_now)
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
        store=store,
        clock=lambda: runtime_now,
        budgets=LoopBudgets(task_timeout_seconds=0.08),
        lease_duration_seconds=0.02,
    )

    with pytest.raises(TaskExecutionTimeout) as timeout:
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    lease = snapshot.task_leases[0]
    assert len(executor.requests) == 1
    assert 0 < timeout.value.timeout_seconds < 0.02
    assert lease.started_at is not None
    assert lease.released_at is not None
    assert lease.release_reason == "task_timeout"


async def test_expired_fence_after_delayed_start_ack_blocks_executor_io() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    executor = HangingExecutor("fake.read.executor", candidate)
    store = DelayedFencedStartAcknowledgementStore(delay_seconds=0.03)
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
        store=store,
        clock=lambda: datetime.now(timezone.utc),
        budgets=LoopBudgets(task_timeout_seconds=0.08),
        lease_duration_seconds=0.02,
    )

    with pytest.raises(OperationStateError, match="lease expired before executor"):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    task = snapshot.tasks[0]
    lease = snapshot.task_leases[0]
    assert executor.requests == []
    assert task.status is TaskStatus.RUNNING
    assert task.evidence_ids == ()
    assert lease.started_at is not None
    assert lease.released_at is None
    assert snapshot.evidence == ()
    assert not any(
        event.type in {"executor.completed", "evidence.accepted", "task.succeeded"}
        for event in snapshot.events
        if event.task_id == task.id
    )


@pytest.mark.parametrize(
    ("idempotent", "replay_safe"),
    ((False, False), (True, True)),
)
async def test_side_effect_timeout_preserves_recovery_classification(
    idempotent: bool,
    replay_safe: bool,
) -> None:
    candidate = EvidenceCandidate(
        kind="fake.write.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    executor = HangingExecutor("fake.write.executor", candidate)
    unsafe_capability = replace(
        _capability(
            capability_id="fake.write",
            executor_id="fake.write.executor",
            evidence_kind="fake.write.result",
        ),
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=idempotent,
        replay_safe=replay_safe,
    )
    registry = CapabilityRegistry(
        capabilities=(unsafe_capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="write_fake",
                capability_id=unsafe_capability.id,
                description="Exercise an unsafe test-owned side effect.",
            ),
        ),
    )
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
        registry_override=registry,
        budgets=LoopBudgets(task_timeout_seconds=0.01),
        tool_calls=(
            ToolCall(
                id="call-1",
                name="write_fake",
                arguments={"key": "alpha"},
            ),
        ),
    )

    with pytest.raises(TaskExecutionTimeout):
        await runtime.submit(
            _proposal(
                operation_id,
                turn_id,
                capability_id="fake.write",
            )
        )

    snapshot = await runtime.inspect(operation_id)
    task = snapshot.tasks[0]
    lease = snapshot.task_leases[0]
    assert len(executor.requests) == 1
    assert task.status is TaskStatus.RUNNING
    assert task.cancellation_requested is False
    assert task.error_code is None
    assert task.evidence_ids == ()
    assert task.execution_facts.replay_safe is replay_safe
    assert (task.execution_facts.idempotency_key is not None) is replay_safe
    assert lease.started_at is not None
    assert lease.released_at is None
    assert snapshot.evidence == ()
    assert not any(
        event.type in {"task.failed", "executor.failed"}
        for event in snapshot.events
        if event.task_id == task.id
    )
    assert snapshot.events[-1].type == "task.outcome_unknown"
    assert snapshot.events[-1].payload["reason"] == "task_timeout"


async def test_expired_holder_cannot_annotate_unknown_side_effect_outcome() -> None:
    candidate = EvidenceCandidate(
        kind="fake.write.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    store = InMemoryOperationStore()
    executor = SlowCancellationSuppressingExecutor(
        "fake.write.executor",
        candidate,
        store,
        suppression_seconds=0.035,
    )
    capability = replace(
        _capability(
            capability_id="fake.write",
            executor_id="fake.write.executor",
            evidence_kind="fake.write.result",
        ),
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=False,
        replay_safe=False,
    )
    registry = CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="write_fake",
                capability_id=capability.id,
                description="Exercise an expired side-effect fence.",
            ),
        ),
    )
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
        store=store,
        registry_override=registry,
        clock=lambda: datetime.now(timezone.utc),
        budgets=LoopBudgets(task_timeout_seconds=0.08),
        lease_duration_seconds=0.04,
        tool_calls=(
            ToolCall(
                id="call-1",
                name="write_fake",
                arguments={"key": "alpha"},
            ),
        ),
    )

    with pytest.raises(TaskExecutionTimeout):
        await runtime.submit(
            _proposal(
                operation_id,
                turn_id,
                capability_id="fake.write",
            )
        )

    assert executor.at_entry is not None
    assert await store.load(operation_id) == executor.at_entry
    snapshot = executor.at_entry.snapshot
    assert len(executor.requests) == 1
    assert snapshot.tasks[0].status is TaskStatus.RUNNING
    assert snapshot.task_leases[0].released_at is None
    assert snapshot.evidence == ()
    assert not any(event.type == "task.outcome_unknown" for event in snapshot.events)


async def test_runtime_fenced_lifecycle_round_trips_through_sqlite(
    tmp_path: Path,
) -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    path = tmp_path / "runtime-fenced.db"
    store = await SQLiteOperationStore.open(path, clock=lambda: NOW)
    operation_id = ""
    final: OperationSnapshot | None = None
    try:
        executor = InspectingExecutor("fake.read.executor", candidate, store)
        runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
            candidate,
            executor_override=executor,
            store=store,
        )

        evidence = await runtime.submit(_proposal(operation_id, turn_id))
        final = await runtime.inspect(operation_id)

        assert final.tasks[0].status is TaskStatus.SUCCEEDED
        assert final.tasks[0].evidence_ids == (evidence.id,)
        assert final.task_leases[0].started_at is not None
        assert final.task_leases[0].released_at is not None
        assert final.task_leases[0].release_reason == "completed"
    finally:
        await store.close()

    assert final is not None
    reopened = await SQLiteOperationStore.open(path)
    try:
        assert (await reopened.load(operation_id)).snapshot == final
    finally:
        await reopened.close()


async def test_idempotent_side_effect_receives_the_persisted_stable_key() -> None:
    capability = replace(
        _capability(
            capability_id="fake.write",
            evidence_kind="fake.write.result",
        ),
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.MEDIUM,
        side_effecting=True,
    )
    candidate = EvidenceCandidate(
        kind="fake.write.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    executor = CandidateExecutor("fake.read.executor", candidate)
    registry = CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="write_fake",
                capability_id=capability.id,
                description="Write one fake value.",
            ),
        ),
    )
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
        registry_override=registry,
        tool_calls=(
            ToolCall(
                id="call-1",
                name="write_fake",
                arguments={"key": "alpha"},
            ),
        ),
    )

    await runtime.submit(
        _proposal(
            operation_id,
            turn_id,
            capability_id=capability.id,
        )
    )

    request = executor.requests[0]
    assert request.idempotency_key == f"{operation_id}:{request.task_id}"
    final = await runtime.inspect(operation_id)
    assert final.tasks[0].execution_facts.idempotency_key == request.idempotency_key


async def test_fenced_start_failure_leaves_claim_without_executor_io() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    store = RejectingFencedStartStore(clock=lambda: NOW)
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate, store=store
    )

    with pytest.raises(OperationStateError, match="fenced"):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    assert executor.requests == []
    assert snapshot.tasks[0].status is TaskStatus.CLAIMED
    assert len(snapshot.task_leases) == 1
    assert snapshot.task_leases[0].started_at is None
    assert snapshot.task_leases[0].released_at is None
    assert [event.type for event in snapshot.events if event.task_id] == [
        "task.created",
        "task.ready",
        "task.claimed",
    ]


async def test_execution_identity_drift_after_readiness_blocks_executor_io() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    executor = CandidateExecutor("fake.read.executor", candidate)
    store = DriftAfterReadyStore(
        lambda: setattr(executor, "executor_id", "drifted.executor")
    )
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
        store=store,
    )

    with pytest.raises(OperationStateError, match="identity"):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    assert executor.requests == []
    assert snapshot.tasks[0].status is TaskStatus.READY
    assert snapshot.task_leases == ()
    assert snapshot.evidence == ()
    assert snapshot.events[-1].type == "task.ready"


async def test_persisted_capability_drift_after_readiness_blocks_executor_io() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    executor = CandidateExecutor("fake.read.executor", candidate)

    class DriftingRegistry(CapabilityRegistry):
        def __init__(self) -> None:
            super().__init__(
                capabilities=(_capability(),),
                executors=(executor,),
                tool_views=(
                    ToolView(
                        name="read_fake",
                        capability_id="fake.read",
                        description="Read one fake value.",
                    ),
                ),
            )
            self.drifted = False

        def resolve_execution(
            self,
            capability_id: str,
        ) -> tuple[Capability, Executor]:
            capability, registered_executor = super().resolve_execution(capability_id)
            if self.drifted:
                capability = replace(
                    capability,
                    risk=RiskLevel.MEDIUM,
                )
            return capability, registered_executor

    registry = DriftingRegistry()
    store = DriftAfterReadyStore(lambda: setattr(registry, "drifted", True))
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
        store=store,
        registry_override=registry,
    )

    with pytest.raises(OperationStateError, match="persisted execution facts"):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    assert executor.requests == []
    assert snapshot.tasks[0].status is TaskStatus.READY
    assert snapshot.task_leases == ()
    assert snapshot.evidence == ()
    assert snapshot.events[-1].type == "task.ready"


async def test_stale_terminal_fence_has_exactly_zero_checkpoint_delta() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    store = RejectingTerminalFenceStore()
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate, store=store
    )

    with pytest.raises(OperationStateError, match="fenced") as caught:
        await runtime.submit(_proposal(operation_id, turn_id))

    assert isinstance(caught.value.__cause__, StaleTaskFenceError)
    assert len(executor.requests) == 1
    assert store.before_rejection is not None
    assert await store.load(operation_id) == store.before_rejection
    snapshot = store.before_rejection.snapshot
    assert snapshot.tasks[0].status is TaskStatus.RUNNING
    assert snapshot.task_leases[0].started_at is not None
    assert snapshot.task_leases[0].released_at is None
    assert snapshot.evidence == ()
    assert snapshot.events[-1].type == "executor.started"


async def _runtime_with_committed_text_response(
    text: str,
) -> tuple[OperationRuntime, str]:
    runtime = OperationRuntime(clock=lambda: NOW)
    started = await runtime.begin(
        AgentTrigger(
            id="text-trigger-1",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "answer"},
            created_at=NOW,
        )
    )
    turn = await runtime.begin_turn(started.operation.id)
    request = ModelRequest(
        operation_id=started.operation.id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=started.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Answer."),),
            ),
        ),
    )
    model_call = await runtime.begin_model_call(
        started.operation.id,
        turn.id,
        "mock:scripted",
        request,
    )
    await runtime.record_model_response(
        started.operation.id,
        model_call.id,
        ModelResponse(text=text, finish_reason=FinishReason.STOP),
        next_phase=LoopPhase.SYNTHESIZING,
    )
    return runtime, started.operation.id


@pytest.mark.parametrize("forgery", ["call_id", "capability", "arguments"])
async def test_submit_rejects_proposal_not_bound_to_committed_tool_call(
    forgery: str,
) -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate, include_other=True
    )
    before = await runtime.inspect(operation_id)
    call_id = "call-1"
    capability_id = "fake.read"
    arguments: Mapping[str, object] = {"key": "alpha"}
    if forgery == "call_id":
        call_id = "call-forged"
    elif forgery == "capability":
        capability_id = "fake.other"
    else:
        arguments = {"key": "beta"}

    with pytest.raises(OperationStateError, match="tool call|proposal"):
        await runtime.submit(
            _proposal(
                operation_id,
                turn_id,
                call_id=call_id,
                capability_id=capability_id,
                arguments=arguments,
            )
        )

    after = await runtime.inspect(operation_id)
    assert after == before
    assert not after.tasks
    assert not after.evidence
    assert executor.requests == []


@pytest.mark.parametrize("forgery", ["description", "input-schema"])
async def test_submit_rejects_forged_committed_tool_projection(
    forgery: str,
) -> None:
    capability = _capability()
    description = "Read one fake value."
    input_schema: Mapping[str, object] = capability.input_schema
    if forgery == "description":
        description = "Forged description with the same tool name."
    else:
        input_schema = {
            "type": "object",
            "properties": {"key": {"type": "integer"}},
            "required": ["key"],
            "additionalProperties": False,
        }
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        request_tools=(
            ToolDefinition(
                name="read_fake",
                description=description,
                input_schema=input_schema,
            ),
        ),
    )
    before = await runtime.inspect(operation_id)

    with pytest.raises(OperationStateError, match="projection|definition|exposed"):
        await runtime.submit(_proposal(operation_id, turn_id))

    after = await runtime.inspect(operation_id)
    assert after == before
    assert after.tasks == ()
    assert after.evidence == ()
    assert executor.requests == []


async def test_submit_rejects_a_later_tool_call_before_its_predecessor() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        tool_calls=(
            ToolCall(
                id="call-1",
                name="read_fake",
                arguments={"key": "alpha"},
            ),
            ToolCall(
                id="call-2",
                name="read_fake",
                arguments={"key": "beta"},
            ),
        ),
    )
    before = await runtime.inspect(operation_id)

    with pytest.raises(OperationStateError, match="sequential order"):
        await runtime.submit(
            _proposal(
                operation_id,
                turn_id,
                call_id="call-2",
                arguments={"key": "beta"},
            )
        )

    after = await runtime.inspect(operation_id)
    assert after == before
    assert after.tasks == ()
    assert after.evidence == ()
    assert executor.requests == []


async def test_call_id_may_be_reused_by_a_later_model_response() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, executor, operation_id, first_turn_id = (
        await _runtime_with_committed_tool_call(candidate)
    )
    first_evidence = await runtime.submit(_proposal(operation_id, first_turn_id))
    await runtime.append_observation(
        Observation(
            operation_id=operation_id,
            turn_id=first_turn_id,
            code="fake.read.succeeded",
            message="First fake read completed.",
            payload=first_evidence.payload,
            success=True,
            created_at=NOW,
            task_id=first_evidence.task_id,
            evidence_id=first_evidence.id,
        )
    )
    after_first = await runtime.inspect(operation_id)
    second_turn = await runtime.begin_turn(operation_id)
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=second_turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=operation_id,
                turn_id=second_turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Read beta."),),
            ),
        ),
        tools=after_first.model_calls[-1].request.tools,
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        second_turn.id,
        "mock:scripted",
        request,
    )
    await runtime.record_model_response(
        operation_id,
        model_call.id,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="read_fake",
                    arguments={"key": "beta"},
                ),
            ),
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )

    second_evidence = await runtime.submit(
        _proposal(
            operation_id,
            second_turn.id,
            call_id="call-1",
            arguments={"key": "beta"},
        )
    )

    final = await runtime.inspect(operation_id)
    assert len(executor.requests) == 2
    assert len(final.tasks) == 2
    assert [task.call_id for task in final.tasks] == ["call-1", "call-1"]
    assert [task.turn_id for task in final.tasks] == [
        first_turn_id,
        second_turn.id,
    ]
    assert final.tasks[0].id != final.tasks[1].id
    assert all(task.status is TaskStatus.SUCCEEDED for task in final.tasks)
    assert len(final.evidence) == 2
    assert first_evidence.task_id != second_evidence.task_id


async def test_readiness_event_construction_failure_retains_the_pending_task() -> None:
    counter = 0
    task_id_allocated = False
    events_after_task = 0

    def fail_on_readiness_event(prefix: str) -> str:
        nonlocal counter, task_id_allocated, events_after_task
        counter += 1
        if prefix == "task":
            task_id_allocated = True
        elif task_id_allocated and prefix == "event":
            events_after_task += 1
            if events_after_task == 2:
                raise RuntimeError("injected task.ready event failure")
        return f"{prefix}-{counter}"

    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        id_factory=fail_on_readiness_event,
    )

    with pytest.raises(RuntimeError, match="task.ready event failure"):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].status is TaskStatus.PENDING
    assert snapshot.tasks[0].call_id == "call-1"
    assert snapshot.events[-1].type == "task.created"
    assert "executor.started" not in [event.type for event in snapshot.events]
    assert snapshot.evidence == ()
    assert executor.requests == []


async def test_executor_identity_mutation_fails_the_committed_task() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    executor = IdentityMutatingExecutor("fake.read.executor", candidate)
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        executor_override=executor,
    )

    with pytest.raises(CapabilityExecutionError, match="identity"):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    assert len(executor.requests) == 1
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].status is TaskStatus.FAILED
    assert snapshot.tasks[0].error_code == "execution_identity_changed"
    assert snapshot.evidence == ()
    event_types = [event.type for event in snapshot.events]
    assert "executor.started" in event_types
    assert "task.failed" in event_types
    assert "executor.failed" in event_types
    assert "executor.completed" not in event_types
    assert "evidence.accepted" not in event_types


async def test_terminal_evidence_event_failure_leaves_task_running() -> None:
    execution_state = {"returned": False}
    counter = 0
    terminal_event_count = 0

    class ReturnFlaggingExecutor(CandidateExecutor):
        async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
            self.requests.append(request)
            execution_state["returned"] = True
            return self.candidate

    def fail_during_evidence_events(prefix: str) -> str:
        nonlocal counter, terminal_event_count
        counter += 1
        if execution_state["returned"] and prefix == "event":
            terminal_event_count += 1
            if terminal_event_count == 2:
                raise RuntimeError("injected evidence event commit failure")
        return f"{prefix}-{counter}"

    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    executor = ReturnFlaggingExecutor("fake.read.executor", candidate)
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        id_factory=fail_during_evidence_events,
        executor_override=executor,
    )

    with pytest.raises(RuntimeError, match="evidence event commit failure"):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    assert len(executor.requests) == 1
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].status is TaskStatus.RUNNING
    assert snapshot.evidence == ()
    event_types = [event.type for event in snapshot.events]
    assert event_types[-1] == "executor.started"
    assert "executor.completed" not in event_types
    assert "evidence.accepted" not in event_types
    assert "task.succeeded" not in event_types


async def test_observation_event_failure_preserves_succeeded_task_and_evidence() -> (
    None
):
    counter = 0
    reject_events = False

    def fail_observation_event(prefix: str) -> str:
        nonlocal counter
        counter += 1
        if reject_events and prefix == "event":
            raise RuntimeError("injected observation.recorded commit failure")
        return f"{prefix}-{counter}"

    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, _, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        id_factory=fail_observation_event,
    )
    evidence = await runtime.submit(_proposal(operation_id, turn_id))
    before = await runtime.inspect(operation_id)
    observation = Observation(
        operation_id=operation_id,
        turn_id=turn_id,
        code="fake.read.succeeded",
        message="Fake read completed.",
        payload=evidence.payload,
        success=True,
        created_at=NOW,
        task_id=evidence.task_id,
        evidence_id=evidence.id,
    )
    reject_events = True

    with pytest.raises(RuntimeError, match="observation.recorded commit failure"):
        await runtime.append_observation(observation)

    after = await runtime.inspect(operation_id)
    assert after == before
    assert len(after.tasks) == 1
    assert after.tasks[0].status is TaskStatus.SUCCEEDED
    assert len(after.evidence) == 1
    assert after.observations == ()
    assert "observation.recorded" not in [event.type for event in after.events]


async def test_readiness_text_must_match_the_committed_model_response() -> None:
    runtime, operation_id = await _runtime_with_committed_text_response(
        "Committed answer."
    )
    before = await runtime.inspect(operation_id)
    readiness = Readiness(
        allowed=True,
        code="ready.test",
        message="Test readiness approved.",
        evaluated_at=NOW,
    )

    with pytest.raises(OperationStateError, match="match.*committed"):
        await runtime.record_readiness(operation_id, "Forged answer.", readiness)

    after = await runtime.inspect(operation_id)
    assert after == before
    assert after.readiness == ()


async def test_readiness_rejects_accepted_evidence_without_observation() -> None:
    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, _, operation_id, first_turn_id = await _runtime_with_committed_tool_call(
        candidate
    )
    await runtime.submit(_proposal(operation_id, first_turn_id))
    turn = await runtime.begin_turn(operation_id)
    request = ModelRequest(
        operation_id=operation_id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=operation_id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Answer from the read."),),
            ),
        ),
    )
    model_call = await runtime.begin_model_call(
        operation_id,
        turn.id,
        "mock:scripted",
        request,
    )
    await runtime.record_model_response(
        operation_id,
        model_call.id,
        ModelResponse(text="Final answer.", finish_reason=FinishReason.STOP),
        next_phase=LoopPhase.SYNTHESIZING,
    )
    before = await runtime.inspect(operation_id)

    with pytest.raises(OperationStateError, match="evidence.*observed"):
        await runtime.record_readiness(
            operation_id,
            "Final answer.",
            Readiness(
                allowed=True,
                code="ready.test",
                message="Test readiness approved.",
                evaluated_at=NOW,
            ),
        )

    after = await runtime.inspect(operation_id)
    assert after == before
    assert len(after.evidence) == 1
    assert after.observations == ()
    assert after.readiness == ()


def test_registry_rejects_executor_whose_identity_differs_from_declaration() -> None:
    executor = CandidateExecutor(
        "forged.executor",
        EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={"key": "alpha", "value": "ALPHA"},
        ),
    )

    with pytest.raises(ValueError, match="executor"):
        CapabilityRegistry(
            capabilities=(_capability(),),
            executors=(executor,),
            tool_views=(
                ToolView(
                    name="read_fake",
                    capability_id="fake.read",
                    description="Read one fake value.",
                ),
            ),
        )


@pytest.mark.parametrize(
    "candidate",
    [
        EvidenceCandidate(
            kind="forged.kind",
            schema_version=1,
            payload={"key": "alpha", "value": "ALPHA"},
        ),
        EvidenceCandidate(
            kind="fake.read.result",
            schema_version=2,
            payload={"key": "alpha", "value": "ALPHA"},
        ),
        EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={"key": "alpha", "value": 42},
        ),
    ],
    ids=["kind", "schema-version", "payload"],
)
async def test_invalid_evidence_fails_task_without_accepting_evidence(
    candidate: EvidenceCandidate,
) -> None:
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate
    )

    with pytest.raises(EvidenceValidationError):
        await runtime.submit(_proposal(operation_id, turn_id))

    snapshot = await runtime.inspect(operation_id)
    assert len(executor.requests) == 1
    assert len(snapshot.tasks) == 1
    assert snapshot.tasks[0].status is TaskStatus.FAILED
    assert snapshot.tasks[0].error_code == "evidence_rejected"
    assert snapshot.evidence == ()
    event_types = [event.type for event in snapshot.events]
    assert "task.failed" in event_types
    assert "executor.failed" in event_types
    assert "evidence.accepted" not in event_types
    assert "task.succeeded" not in event_types
