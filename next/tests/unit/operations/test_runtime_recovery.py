from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from daita._json import canonical_json
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
    EvidenceCandidate,
    ExecutionRequest,
    Executor,
    RiskLevel,
    ToolView,
)
from daita.events.models import RuntimeEvent
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.loop.models import LoopBudgets, LoopPhase, LoopState, Turn
from daita.operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from daita.operations.leases import TaskLease
from daita.operations.models import (
    AgentTrigger,
    Evidence,
    Operation,
    OperationStatus,
    Task,
    TaskExecutionFacts,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime
from daita.operations.store import InMemoryOperationStore

NOW = datetime(2026, 7, 18, 9, 0, tzinfo=timezone.utc)
OPERATION_ID = "operation-recovery"
TASK_ID = "task-recovery"
TURN_ID = "turn-recovery"
CALL_ID = "call-recovery"


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current


class RecordingExecutor:
    def __init__(self, candidate: EvidenceCandidate) -> None:
        self.executor_id = "fake.executor"
        self.candidate = candidate
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        return self.candidate


class LookupRecordingRegistry(CapabilityRegistry):
    def __init__(
        self,
        capability: Capability,
        executor: RecordingExecutor,
    ) -> None:
        super().__init__(
            capabilities=(capability,),
            executors=(executor,),
            tool_views=(
                ToolView(
                    name="fake_action",
                    capability_id=capability.id,
                    description="Exercise one persisted recovery task.",
                ),
            ),
        )
        self.execution_lookups: list[str] = []

    def resolve_execution(self, capability_id: str) -> tuple[Capability, Executor]:
        self.execution_lookups.append(capability_id)
        return super().resolve_execution(capability_id)


def _content_hash(payload: object) -> str:
    return (
        "sha256:" + hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()
    )


def _capability(*, side_effecting: bool) -> Capability:
    return Capability(
        id="fake.write" if side_effecting else "fake.read",
        owner="recovery-tests",
        description="Exercise fail-closed task recovery.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="fake.result",
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
        executor_id="fake.executor",
        access_mode=AccessMode.WRITE if side_effecting else AccessMode.READ,
        risk=RiskLevel.HIGH if side_effecting else RiskLevel.LOW,
        side_effecting=side_effecting,
        idempotent=not side_effecting,
        replay_safe=not side_effecting,
    )


def _replay_safe_side_effect() -> Capability:
    return replace(
        _capability(side_effecting=True),
        idempotent=True,
        replay_safe=True,
    )


def _registry(
    capability: Capability,
    executor: RecordingExecutor,
) -> LookupRecordingRegistry:
    return LookupRecordingRegistry(capability, executor)


def _execution_facts(capability: Capability) -> TaskExecutionFacts:
    arguments = {"key": "alpha"}
    return TaskExecutionFacts(
        capability_fingerprint=capability.contract_fingerprint,
        arguments_hash=_content_hash(arguments),
        access_mode=capability.access_mode,
        risk=capability.risk,
        side_effecting=capability.side_effecting,
        idempotent=capability.idempotent,
        replay_safe=capability.replay_safe,
        idempotency_key=(
            f"{OPERATION_ID}:{TASK_ID}"
            if capability.side_effecting and capability.idempotent
            else None
        ),
    )


def _task(
    capability: Capability,
    *,
    status: TaskStatus,
    updated_at: datetime = NOW,
) -> Task:
    evidence_ids = ("evidence-existing",) if status is TaskStatus.SUCCEEDED else ()
    return Task(
        id=TASK_ID,
        operation_id=OPERATION_ID,
        turn_id=TURN_ID,
        call_id=CALL_ID,
        capability_id=capability.id,
        executor_id=capability.executor_id,
        status=status,
        attempt=1,
        arguments={"key": "alpha"},
        execution_facts=_execution_facts(capability),
        evidence_ids=evidence_ids,
        error_code="task_failed" if status is TaskStatus.FAILED else None,
        manual_recovery_reason=(
            "operator_review_required"
            if status is TaskStatus.MANUAL_RECOVERY_REQUIRED
            else None
        ),
        created_at=NOW,
        updated_at=updated_at,
    )


def _accepted_evidence(task: Task) -> Evidence:
    payload = {"key": "alpha", "value": "EXISTING"}
    return Evidence(
        id=task.evidence_ids[0],
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
        created_at=task.updated_at,
    )


def _snapshot(
    capability: Capability,
    *,
    status: TaskStatus,
    expired_running: bool = False,
) -> OperationSnapshot:
    started_at = NOW + timedelta(seconds=1)
    updated_at = started_at if expired_running else NOW
    task = _task(capability, status=status, updated_at=updated_at)
    trigger = AgentTrigger(
        id="trigger-recovery",
        agent_id="agent-recovery",
        kind=TriggerKind.USER,
        source_id="user-recovery",
        session_id="session-recovery",
        payload={"message": "recover alpha"},
        created_at=NOW,
    )
    operation = Operation(
        id=OPERATION_ID,
        agent_id=trigger.agent_id,
        trigger_id=trigger.id,
        session_id=trigger.session_id,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=updated_at,
    )
    request = ModelRequest(
        operation_id=operation.id,
        turn_id=TURN_ID,
        messages=(
            CanonicalMessage(
                agent_id=operation.agent_id,
                operation_id=operation.id,
                session_id=operation.session_id,
                turn_id=TURN_ID,
                role=MessageRole.USER,
                content=(TextBlock("Recover alpha."),),
            ),
        ),
        tools=(),
    )
    response = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(
            ToolCall(
                id=CALL_ID,
                name="fake_action",
                arguments=task.arguments,
            ),
        ),
    )
    model_call = ModelCall(
        id="model-call-recovery",
        operation_id=operation.id,
        turn_id=TURN_ID,
        provider_id="mock:scripted",
        request=request,
        response=response,
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )
    turn = Turn(
        id=TURN_ID,
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
    leases = (
        (
            TaskLease(
                operation_id=operation.id,
                task_id=task.id,
                attempt=1,
                fencing_token=1,
                holder_id="expired-holder",
                acquired_at=NOW,
                expires_at=NOW + timedelta(seconds=10),
                started_at=started_at,
            ),
        )
        if expired_running
        else ()
    )
    evidence = (_accepted_evidence(task),) if status is TaskStatus.SUCCEEDED else ()
    return OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.AWAITING_EXECUTION,
            turn_count=1,
            action_count=1,
        ),
        budgets=LoopBudgets(),
        turns=(turn,),
        model_calls=(model_call,),
        readiness=(),
        tasks=(task,),
        task_dependencies=(),
        task_leases=leases,
        evidence=evidence,
        observations=(),
        events=events,
    )


async def _seeded_runtime(
    capability: Capability,
    *,
    status: TaskStatus,
    expired_running: bool = False,
) -> tuple[
    OperationRuntime,
    RecordingExecutor,
    InMemoryOperationStore,
    LookupRecordingRegistry,
]:
    candidate = EvidenceCandidate(
        kind="fake.result",
        schema_version=1,
        payload={"key": "alpha", "value": "RECOVERED"},
    )
    executor = RecordingExecutor(candidate)
    registry = _registry(capability, executor)
    clock = MutableClock(NOW + timedelta(seconds=10))
    store = InMemoryOperationStore(clock=clock)
    await store.create(
        _snapshot(
            capability,
            status=status,
            expired_running=expired_running,
        )
    )
    runtime = OperationRuntime(
        capabilities=registry,
        clock=clock,
        store=store,
        lease_holder_id="recovery-holder",
        lease_duration_seconds=20,
    )
    return runtime, executor, store, registry


@pytest.mark.parametrize(
    ("capability_factory", "expected_idempotency_key"),
    (
        (lambda: _capability(side_effecting=False), None),
        (_replay_safe_side_effect, f"{OPERATION_ID}:{TASK_ID}"),
    ),
    ids=("replay_safe_read", "keyed_side_effect"),
)
async def test_resume_reclaims_expired_replay_safe_running_task_with_same_identity(
    capability_factory: Callable[[], Capability],
    expected_idempotency_key: str | None,
) -> None:
    capability = capability_factory()
    runtime, executor, _, _ = await _seeded_runtime(
        capability,
        status=TaskStatus.RUNNING,
        expired_running=True,
    )

    evidence = await runtime.resume_task(OPERATION_ID, TASK_ID)

    assert evidence is not None
    assert evidence.task_id == TASK_ID
    assert evidence.attempt == 2
    assert len(executor.requests) == 1
    request = executor.requests[0]
    assert request.operation_id == OPERATION_ID
    assert request.task_id == TASK_ID
    assert request.capability_id == capability.id
    assert request.executor_id == capability.executor_id
    assert request.arguments.to_dict() == {"key": "alpha"}
    assert request.idempotency_key == expected_idempotency_key
    assert request.attempt == 2
    assert request.fencing_token == 2

    final = await runtime.inspect(OPERATION_ID)
    task = final.tasks[0]
    assert task.status is TaskStatus.SUCCEEDED
    assert task.id == TASK_ID
    assert task.attempt == 2
    assert task.execution_facts.idempotency_key == expected_idempotency_key
    assert task.evidence_ids == (evidence.id,)
    assert len(final.task_leases) == 2
    assert final.task_leases[0].release_reason == "expired_replay_safe"
    assert final.task_leases[1].attempt == 2
    assert final.task_leases[1].fencing_token == 2
    assert final.task_leases[1].release_reason == "completed"


async def test_resume_classifies_expired_unsafe_started_side_effect_as_manual() -> None:
    capability = _capability(side_effecting=True)
    runtime, executor, _, _ = await _seeded_runtime(
        capability,
        status=TaskStatus.RUNNING,
        expired_running=True,
    )

    evidence = await runtime.resume_task(OPERATION_ID, TASK_ID)

    assert evidence is None
    assert executor.requests == []
    final = await runtime.inspect(OPERATION_ID)
    task = final.tasks[0]
    assert task.status is TaskStatus.MANUAL_RECOVERY_REQUIRED
    assert task.manual_recovery_reason == "unknown_side_effect_outcome"
    assert task.attempt == 1
    assert task.evidence_ids == ()
    assert final.evidence == ()
    assert len(final.task_leases) == 1
    assert final.task_leases[0].released_at == NOW + timedelta(seconds=10)
    assert final.task_leases[0].release_reason == "expired_unknown_outcome"
    assert final.events[-1].type == "task.lease_lost"
    assert final.events[-1].payload["to_status"] == (
        TaskStatus.MANUAL_RECOVERY_REQUIRED.value
    )


async def test_resume_returns_existing_success_without_reexecuting() -> None:
    capability = _capability(side_effecting=False)
    runtime, executor, store, registry = await _seeded_runtime(
        capability,
        status=TaskStatus.SUCCEEDED,
    )
    before = await store.load(OPERATION_ID)

    evidence = await runtime.resume_task(OPERATION_ID, TASK_ID)

    assert evidence is not None
    assert evidence.id == "evidence-existing"
    assert evidence.payload.to_dict() == {"key": "alpha", "value": "EXISTING"}
    assert executor.requests == []
    final = await runtime.inspect(OPERATION_ID)
    assert final.tasks[0].status is TaskStatus.SUCCEEDED
    assert final.tasks[0].evidence_ids == (evidence.id,)
    assert final.evidence == (evidence,)
    assert await store.load(OPERATION_ID) == before
    assert registry.execution_lookups == []


@pytest.mark.parametrize(
    "status",
    (
        TaskStatus.FAILED,
        TaskStatus.CANCELLED,
        TaskStatus.MANUAL_RECOVERY_REQUIRED,
    ),
)
async def test_resume_rejects_terminal_non_success_without_reexecuting(
    status: TaskStatus,
) -> None:
    capability = _capability(side_effecting=False)
    runtime, executor, store, registry = await _seeded_runtime(
        capability,
        status=status,
    )
    before = await store.load(OPERATION_ID)

    evidence = await runtime.resume_task(OPERATION_ID, TASK_ID)

    assert evidence is None
    assert executor.requests == []
    assert registry.execution_lookups == []
    assert await store.load(OPERATION_ID) == before
    final = await runtime.inspect(OPERATION_ID)
    assert final.tasks[0].status is status
    assert final.evidence == ()
