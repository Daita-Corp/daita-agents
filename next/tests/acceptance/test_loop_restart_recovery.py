from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityExecutionError,
    CapabilityRegistry,
    EvidenceArtifact,
    EvidenceCandidate,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.driver import AgentLoop
from daita.loop.models import (
    LoopBudgets,
    LoopExit,
    LoopExitKind,
    LoopPhase,
    Readiness,
    Turn,
)
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.leases import TaskClaimRequest, TaskLeaseGuard
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    Task,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime, OperationStateError
from daita.operations.store import (
    CommitResult,
    TaskClaimResult,
    VersionedOperation,
)
from daita.storage.blobs import BlobMetadata, BlobPut, LocalBlobStore
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 16, 20, 0, tzinfo=timezone.utc)


def _capability() -> Capability:
    return Capability(
        id="fake.read",
        owner="restart-test",
        description="Read one deterministic value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="fake.read.result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        executor_id="fake.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


class RecordingExecutor:
    executor_id = "fake.read.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        key = request.arguments["key"]
        assert isinstance(key, str)
        return EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={"key": key, "value": key.upper()},
        )


class AbruptProcessExit(BaseException):
    """Test-only process loss that bypasses ordinary exception cleanup."""


class ProcessExitExecutor(RecordingExecutor):
    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        raise AbruptProcessExit


class FailingExecutor(RecordingExecutor):
    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        raise RuntimeError("injected executor failure")


class ArtifactExecutor(RecordingExecutor):
    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        key = request.arguments["key"]
        assert isinstance(key, str)
        return EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={"key": key, "value": key.upper()},
            artifact=EvidenceArtifact(
                content=f'{{"key":"{key}","value":"{key.upper()}"}}'.encode(),
                media_type="application/json",
                sensitivity_class="internal",
                retention_class="operation",
            ),
        )


class CrashAfterCommittedEventStore:
    """Raise process loss only after the SQLite delegate commits one event."""

    def __init__(self, delegate: SQLiteOperationStore, event_type: str) -> None:
        self.delegate = delegate
        self.event_type = event_type
        self.crashed = False

    def _after_commit(self, result: CommitResult) -> None:
        if self.crashed or not any(
            event.type == self.event_type for event in result.committed_events
        ):
            return
        self.crashed = True
        raise AbruptProcessExit

    async def create(self, snapshot: OperationSnapshot) -> CommitResult:
        result = await self.delegate.create(snapshot)
        self._after_commit(result)
        return result

    async def load(self, operation_id: str) -> VersionedOperation:
        return await self.delegate.load(operation_id)

    async def load_nonterminal(
        self,
        agent_id: str,
    ) -> tuple[VersionedOperation, ...]:
        return await self.delegate.load_nonterminal(agent_id)

    async def load_by_trigger(
        self,
        trigger_id: str,
    ) -> VersionedOperation | None:
        return await self.delegate.load_by_trigger(trigger_id)

    async def commit(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
    ) -> CommitResult:
        result = await self.delegate.commit(
            snapshot,
            expected_revision=expected_revision,
        )
        self._after_commit(result)
        return result

    async def claim_task(
        self,
        request: TaskClaimRequest,
        *,
        expected_revision: int,
    ) -> TaskClaimResult:
        result = await self.delegate.claim_task(
            request,
            expected_revision=expected_revision,
        )
        self._after_commit(result.commit_result)
        return result

    async def renew_task_lease(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
        lease_duration_seconds: float,
    ) -> CommitResult:
        result = await self.delegate.renew_task_lease(
            snapshot,
            expected_revision=expected_revision,
            guard=guard,
            lease_duration_seconds=lease_duration_seconds,
        )
        self._after_commit(result)
        return result

    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        result = await self.delegate.commit_fenced(
            snapshot,
            expected_revision=expected_revision,
            guard=guard,
        )
        self._after_commit(result)
        return result

    async def recover_expired_task(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        result = await self.delegate.recover_expired_task(
            snapshot,
            expected_revision=expected_revision,
            guard=guard,
        )
        self._after_commit(result)
        return result


class CrashBeforeModelResponseRuntime(OperationRuntime):
    """Lose one provider result while its STARTED request remains durable."""

    crashed = False

    async def record_model_response(
        self,
        operation_id: str,
        model_call_id: str,
        response: ModelResponse,
        *,
        next_phase: LoopPhase,
    ) -> None:
        if not self.crashed:
            self.crashed = True
            raise AbruptProcessExit
        await super().record_model_response(
            operation_id,
            model_call_id,
            response,
            next_phase=next_phase,
        )


class CrashAfterPutBlobStore(LocalBlobStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.committed: BlobMetadata | None = None

    async def put(self, request: BlobPut, content: bytes) -> BlobMetadata:
        self.committed = await super().put(request, content)
        raise AbruptProcessExit


class MutableClock:
    def __init__(self, current: datetime) -> None:
        self.current = current

    def __call__(self) -> datetime:
        return self.current


class NamespacedIds:
    def __init__(self, namespace: str) -> None:
        self.namespace = namespace
        self.counts: dict[str, int] = {}

    def __call__(self, prefix: str) -> str:
        self.counts[prefix] = self.counts.get(prefix, 0) + 1
        return f"{prefix}-{self.namespace}-{self.counts[prefix]}"


class MaterializeOnlyRuntime(OperationRuntime):
    async def materialize_only(self, proposal: ActionProposal) -> Task:
        return await self._materialize_task(proposal)


def _registry(
    executor: RecordingExecutor,
    capability: Capability | None = None,
) -> CapabilityRegistry:
    capability = _capability() if capability is None else capability
    return CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_fake_value",
                capability_id=capability.id,
                description="Read one fake value by key.",
            ),
        ),
    )


class CountingDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry
        self.validation_calls = 0
        self.projection_calls = 0
        self.readiness_calls = 0

    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        assert operation.operation.status is OperationStatus.RUNNING
        return self.registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        self.validation_calls += 1
        view, capability = self.registry.resolve_tool(call.name)
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=view.capability_id,
            arguments=self.registry.validate_arguments(
                capability.id,
                call.arguments,
            ),
            proposed_at=NOW,
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        self.projection_calls += 1
        return Observation(
            operation_id=evidence.operation_id,
            turn_id=evidence.turn_id,
            code="fake.read.succeeded",
            message="Fake read completed.",
            payload=evidence.payload,
            success=True,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            created_at=NOW,
        )

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        self.readiness_calls += 1
        assert text == "Recovered answer."
        assert operation.evidence
        assert len(operation.evidence) == len(operation.observations)
        return Readiness(
            allowed=True,
            code="ready.recovered",
            message="Recovered evidence is ready.",
            evaluated_at=NOW,
        )


class TextDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry
        self.tool_view_calls = 0
        self.readiness_calls = 0

    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        self.tool_view_calls += 1
        return self.registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("text-only recovery cannot validate an action")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("text-only recovery cannot project evidence")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        self.readiness_calls += 1
        assert text == "Recovered answer."
        return Readiness(
            allowed=True,
            code="ready.recovered_text",
            message="The recovered text is ready.",
            evaluated_at=NOW,
        )


class CountingContextBuilder:
    def __init__(self) -> None:
        self.calls = 0

    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        self.calls += 1
        messages: tuple[CanonicalMessage, ...]
        if not operation.model_calls:
            messages = (
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Read alpha."),),
                ),
            )
        else:
            first_call = operation.model_calls[0]
            assert first_call.response is not None
            messages_list = [
                *first_call.request.messages,
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=first_call.turn_id,
                    role=MessageRole.ASSISTANT,
                    tool_calls=first_call.response.tool_calls,
                ),
            ]
            for observation in operation.observations:
                if observation.task_id is None:
                    continue
                task = next(
                    task for task in operation.tasks if task.id == observation.task_id
                )
                messages_list.append(
                    CanonicalMessage(
                        agent_id=operation.operation.agent_id,
                        operation_id=operation.operation.id,
                        turn_id=first_call.turn_id,
                        role=MessageRole.TOOL,
                        content=(
                            ToolResultBlock(
                                call_id=task.call_id,
                                output=observation.payload,
                            ),
                        ),
                    )
                )
            messages = tuple(messages_list)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=messages,
            tools=tools,
        )


class TextContextBuilder:
    def __init__(self) -> None:
        self.calls = 0

    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        self.calls += 1
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Continue from the committed correction."),),
                ),
            ),
            tools=tools,
        )


def _trigger() -> AgentTrigger:
    return AgentTrigger(
        id="trigger-restart-after-response",
        agent_id="agent-restart",
        kind=TriggerKind.USER,
        source_id="user-restart",
        payload={"key": "alpha"},
        created_at=NOW,
    )


def _final_response() -> ModelResponse:
    return ModelResponse(
        text="Recovered answer.",
        finish_reason=FinishReason.STOP,
        usage=ModelUsage(input_tokens=7, output_tokens=2),
    )


def _tool_response() -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-alpha",
                name="read_fake_value",
                arguments={"key": "alpha"},
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=ModelUsage(input_tokens=5, output_tokens=3),
    )


def _two_tool_response() -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            ToolCall(
                id="call-alpha",
                name="read_fake_value",
                arguments={"key": "alpha"},
            ),
            ToolCall(
                id="call-beta",
                name="read_fake_value",
                arguments={"key": "beta"},
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=ModelUsage(input_tokens=8, output_tokens=5),
    )


def _three_tool_response() -> ModelResponse:
    return ModelResponse(
        tool_calls=(
            *_two_tool_response().tool_calls,
            ToolCall(
                id="call-gamma",
                name="read_fake_value",
                arguments={"key": "gamma"},
            ),
        ),
        finish_reason=FinishReason.TOOL_CALLS,
        usage=ModelUsage(input_tokens=11, output_tokens=7),
    )


async def _seed_model_call(
    runtime: OperationRuntime,
    registry: CapabilityRegistry,
    response: ModelResponse | None,
    *,
    budgets: LoopBudgets = LoopBudgets(),
    trigger: AgentTrigger | None = None,
) -> tuple[OperationSnapshot, Turn, ModelRequest, str]:
    started = await runtime.begin(
        _trigger() if trigger is None else trigger,
        budgets=budgets,
    )
    turn = await runtime.begin_turn(started.operation.id)
    snapshot = await runtime.inspect(started.operation.id)
    request = await CountingContextBuilder().build(
        snapshot,
        turn,
        registry.tool_definitions(),
    )
    model_call = await runtime.begin_model_call(
        started.operation.id,
        turn.id,
        "mock:scripted",
        request,
    )
    if response is not None:
        await runtime.record_model_response(
            started.operation.id,
            model_call.id,
            response,
            next_phase=(
                LoopPhase.VALIDATING_ACTION
                if response.tool_calls
                else LoopPhase.SYNTHESIZING
            ),
        )
    return started, turn, request, model_call.id


def _proposal(
    operation_id: str,
    turn_id: str,
    *,
    capability_id: str = "fake.read",
    call_id: str = "call-alpha",
    key: str = "alpha",
) -> ActionProposal:
    return ActionProposal(
        operation_id=operation_id,
        turn_id=turn_id,
        call_id=call_id,
        capability_id=capability_id,
        arguments={"key": key},
        proposed_at=NOW,
    )


async def test_resume_reuses_committed_tool_response_after_sqlite_reopen(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "restart.db"
    first_executor = RecordingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        initial_context = CountingContextBuilder()

        started = await first_runtime.begin(_trigger())
        turn = await first_runtime.begin_turn(started.operation.id)
        before_request = await first_runtime.inspect(started.operation.id)
        request = await initial_context.build(
            before_request,
            turn,
            first_registry.tool_definitions(),
        )
        model_call = await first_runtime.begin_model_call(
            started.operation.id,
            turn.id,
            "mock:scripted",
            request,
        )
        committed_response = ModelResponse(
            tool_calls=(
                ToolCall(
                    id="call-alpha",
                    name="read_fake_value",
                    arguments={"key": "alpha"},
                ),
            ),
            finish_reason=FinishReason.TOOL_CALLS,
            usage=ModelUsage(input_tokens=5, output_tokens=3),
        )
        await first_runtime.record_model_response(
            started.operation.id,
            model_call.id,
            committed_response,
            next_phase=LoopPhase.VALIDATING_ACTION,
        )
        before_restart = await first_runtime.inspect(started.operation.id)
        assert before_restart.tasks == ()
        assert before_restart.model_calls[0].response == committed_response
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        resumed_context = CountingContextBuilder()
        resumed_domain = CountingDomain(resumed_registry)
        resumed_provider = MockModelProvider(
            (
                ModelResponse(
                    text="Recovered answer.",
                    finish_reason=FinishReason.STOP,
                    usage=ModelUsage(input_tokens=7, output_tokens=2),
                ),
            )
        )
        resumed_loop = AgentLoop(
            runtime=resumed_runtime,
            model=resumed_provider,
            context_builder=resumed_context,
            domain=resumed_domain,
        )

        result = await resumed_loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert final.operation.status is OperationStatus.SUCCEEDED
    assert final.operation.id == before_restart.operation.id
    assert final.turns[0] == before_restart.turns[0]
    assert final.model_calls[0] == before_restart.model_calls[0]
    assert resumed_context.calls == 1
    assert len(resumed_provider.requests) == 1
    assert resumed_domain.validation_calls == 1
    assert resumed_domain.projection_calls == 1
    assert resumed_domain.readiness_calls == 1
    assert len(resumed_executor.requests) == 1
    assert len(final.tasks) == 1
    assert final.tasks[0].status is TaskStatus.SUCCEEDED
    assert len(final.evidence) == len(final.observations) == 1
    assert [event.type for event in final.events].count("task.created") == 1
    assert [event.type for event in final.events].count("model_response.recorded") == 2
    resumed_provider.assert_consumed()


async def test_resume_reuses_requestless_turn_after_sqlite_reopen(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "requestless-turn.db"
    executor = RecordingExecutor()
    registry = _registry(executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started = await first_runtime.begin(_trigger())
        existing_turn = await first_runtime.begin_turn(started.operation.id)
    finally:
        await first_store.close()

    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        context = CountingContextBuilder()
        domain = TextDomain(registry)
        provider = MockModelProvider((_final_response(),))
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=context,
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert len(final.turns) == 1
    assert final.turns[0].id == existing_turn.id
    assert final.turns[0].number == existing_turn.number
    assert final.turns[0].created_at == existing_turn.created_at
    assert final.model_calls[0].request.turn_id == existing_turn.id
    assert context.calls == 1
    assert domain.tool_view_calls == 1
    assert len(provider.requests) == 1


async def test_resume_resends_exact_started_model_request_after_sqlite_reopen(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "started-model-call.db"
    executor = RecordingExecutor()
    registry = _registry(executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, existing_turn, request, model_call_id = await _seed_model_call(
            first_runtime,
            registry,
            None,
        )
    finally:
        await first_store.close()

    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        context = CountingContextBuilder()
        domain = TextDomain(registry)
        provider = MockModelProvider((_final_response(),))
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=context,
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert len(final.turns) == 1
    assert final.turns[0].id == existing_turn.id
    assert final.turns[0].number == existing_turn.number
    assert final.turns[0].created_at == existing_turn.created_at
    assert len(final.model_calls) == 1
    assert final.model_calls[0].id == model_call_id
    assert final.model_calls[0].request == request
    assert provider.requests == (request,)
    assert context.calls == 0
    assert domain.tool_view_calls == 0


async def test_resume_existing_pending_task_without_revalidation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "pending-task.db"
    first_executor = RecordingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = MaterializeOnlyRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _tool_response(),
        )
        pending = await first_runtime.materialize_only(
            _proposal(started.operation.id, turn.id)
        )
        assert pending.status is TaskStatus.PENDING
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        context = CountingContextBuilder()
        domain = CountingDomain(resumed_registry)
        provider = MockModelProvider((_final_response(),))
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=context,
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert final.tasks[0].id == pending.id
    assert final.tasks[0].status is TaskStatus.SUCCEEDED
    assert domain.validation_calls == 0
    assert domain.projection_calls == 1
    assert len(resumed_executor.requests) == 1
    assert [event.type for event in final.events].count("task.created") == 1


async def test_resume_mixed_multi_call_checkpoints_in_committed_order(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "mixed-multi-call.db"
    first_executor = RecordingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = MaterializeOnlyRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _three_tool_response(),
        )
        first_evidence = await first_runtime.submit(
            _proposal(started.operation.id, turn.id)
        )
        first_observation = await CountingDomain(first_registry).project_observation(
            first_evidence
        )
        await first_runtime.append_observation(first_observation)
        second_task = await first_runtime.materialize_only(
            _proposal(
                started.operation.id,
                turn.id,
                call_id="call-beta",
                key="beta",
            )
        )
        before_restart = await first_runtime.inspect(started.operation.id)
        assert [task.call_id for task in before_restart.tasks] == [
            "call-alpha",
            "call-beta",
        ]
        assert second_task.status is TaskStatus.PENDING
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        domain = CountingDomain(resumed_registry)
        provider = MockModelProvider((_final_response(),))
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=CountingContextBuilder(),
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert [task.call_id for task in final.tasks] == [
        "call-alpha",
        "call-beta",
        "call-gamma",
    ]
    assert final.tasks[0] == before_restart.tasks[0]
    assert final.tasks[1].id == second_task.id
    assert [request.arguments["key"] for request in resumed_executor.requests] == [
        "beta",
        "gamma",
    ]
    assert domain.validation_calls == 1
    assert domain.projection_calls == 2
    assert len(final.evidence) == len(final.observations) == 3
    assert [event.type for event in final.events].count("task.created") == 3
    assert len(provider.requests) == 1


async def test_resume_projects_existing_evidence_without_executor_replay(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "accepted-evidence.db"
    first_executor = RecordingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _tool_response(),
        )
        evidence = await first_runtime.submit(_proposal(started.operation.id, turn.id))
        before_restart = await first_runtime.inspect(started.operation.id)
        assert before_restart.observations == ()
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        domain = CountingDomain(resumed_registry)
        provider = MockModelProvider((_final_response(),))
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=CountingContextBuilder(),
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert final.evidence == (evidence,)
    assert final.observations[0].evidence_id == evidence.id
    assert domain.validation_calls == 0
    assert domain.projection_calls == 1
    assert resumed_executor.requests == []


async def test_resume_projects_only_missing_item_from_plural_task_evidence(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "plural-evidence.db"
    source_executor = RecordingExecutor()
    source_registry = _registry(source_executor)
    source_runtime = OperationRuntime(
        capabilities=source_registry,
        clock=lambda: NOW,
    )
    started, turn, _, _ = await _seed_model_call(
        source_runtime,
        source_registry,
        _tool_response(),
    )
    first_evidence = await source_runtime.submit(
        _proposal(started.operation.id, turn.id)
    )
    first_observation = await CountingDomain(source_registry).project_observation(
        first_evidence
    )
    await source_runtime.append_observation(first_observation)
    source = await source_runtime.inspect(started.operation.id)
    second_evidence = replace(first_evidence, id="evidence-second")
    plural_task = replace(
        source.tasks[0],
        evidence_ids=(first_evidence.id, second_evidence.id),
    )
    crafted = replace(
        source,
        tasks=(plural_task,),
        evidence=(first_evidence, second_evidence),
    )

    seed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        await seed_store.create(crafted)
    finally:
        await seed_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        domain = CountingDomain(resumed_registry)
        provider = MockModelProvider((_final_response(),))
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=CountingContextBuilder(),
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert final.evidence == (first_evidence, second_evidence)
    assert final.observations[0] == first_observation
    assert [observation.evidence_id for observation in final.observations] == [
        first_evidence.id,
        second_evidence.id,
    ]
    assert domain.validation_calls == 0
    assert domain.projection_calls == 1
    assert resumed_executor.requests == []


async def test_resume_live_running_task_returns_waiting_without_mutation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "live-task.db"
    first_executor = ProcessExitExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _tool_response(),
        )
        with pytest.raises(AbruptProcessExit):
            await first_runtime.submit(_proposal(started.operation.id, turn.id))
        before_restart = await first_runtime.inspect(started.operation.id)
        before_versioned = await first_store.load(started.operation.id)
        assert before_restart.tasks[0].status is TaskStatus.RUNNING
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        domain = CountingDomain(resumed_registry)
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider(()),
            context_builder=CountingContextBuilder(),
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        after_resume = await resumed_runtime.inspect(started.operation.id)
        after_versioned = await resumed_store.load(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.WAITING
    assert result.reason == "task_lease_active"
    assert after_resume == before_restart
    assert after_versioned == before_versioned
    assert domain.validation_calls == 0
    assert domain.projection_calls == 0
    assert resumed_executor.requests == []


async def test_resume_failed_task_terminalizes_operation_without_reexecution(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "failed-task.db"
    first_executor = FailingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _tool_response(),
        )
        with pytest.raises(CapabilityExecutionError):
            await first_runtime.submit(_proposal(started.operation.id, turn.id))
        before_restart = await first_runtime.inspect(started.operation.id)
        assert before_restart.tasks[0].status is TaskStatus.FAILED
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        domain = CountingDomain(resumed_registry)
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider(()),
            context_builder=CountingContextBuilder(),
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "action_processing_failed"
    assert final.operation.status is OperationStatus.FAILED
    assert final.tasks[0] == before_restart.tasks[0]
    assert domain.validation_calls == 0
    assert domain.projection_calls == 0
    assert resumed_executor.requests == []


async def test_resume_expired_unsafe_task_waits_for_manual_recovery(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "manual-recovery-task.db"
    clock = MutableClock(NOW)
    unsafe_capability = replace(
        _capability(),
        id="fake.write",
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.HIGH,
        side_effecting=True,
        idempotent=False,
        replay_safe=False,
    )
    first_executor = ProcessExitExecutor()
    first_registry = _registry(first_executor, unsafe_capability)
    first_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=clock,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _tool_response(),
            budgets=LoopBudgets(max_wall_time_seconds=10),
        )
        with pytest.raises(AbruptProcessExit):
            await first_runtime.submit(
                _proposal(
                    started.operation.id,
                    turn.id,
                    capability_id=unsafe_capability.id,
                )
            )
        before_restart = await first_runtime.inspect(started.operation.id)
        assert before_restart.tasks[0].status is TaskStatus.RUNNING
        lease = before_restart.task_leases[0]
    finally:
        await first_store.close()

    clock.current = lease.expires_at + timedelta(microseconds=1)
    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor, unsafe_capability)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=clock,
        )
        domain = CountingDomain(resumed_registry)
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider(()),
            context_builder=CountingContextBuilder(),
            domain=domain,
        )

        recovered = await loop.recover_startup("agent-restart")
        assert tuple(exit.operation_id for exit in recovered) == (started.operation.id,)
        result = recovered[0]
        after_manual = await resumed_store.load(started.operation.id)
        repeated = await loop.recover_startup("agent-restart")
        after_repeated = await resumed_store.load(started.operation.id)
        final = after_manual.snapshot
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.WAITING
    assert result.reason == "manual_recovery_required"
    assert final.operation.status is OperationStatus.RUNNING
    assert final.tasks[0].status is TaskStatus.MANUAL_RECOVERY_REQUIRED
    assert final.tasks[0].manual_recovery_reason == "unknown_side_effect_outcome"
    assert final.task_leases[0].released_at == clock.current
    assert final.task_leases[0].release_reason == "expired_unknown_outcome"
    assert tuple(exit.kind for exit in repeated) == (LoopExitKind.WAITING,)
    assert repeated[0].reason == "manual_recovery_required"
    assert after_repeated == after_manual
    assert not any(event.type == "budget.exhausted" for event in final.events)
    assert not any(event.type == "operation.failed" for event in final.events)
    assert domain.validation_calls == 0
    assert domain.projection_calls == 0
    assert resumed_executor.requests == []


async def test_resume_attributes_aggregate_observation_budget_to_durable_frontier(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "observation-frontier.db"
    first_executor = RecordingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _two_tool_response(),
            budgets=LoopBudgets(max_observation_characters=60),
        )
        domain = CountingDomain(first_registry)
        first_evidence = await first_runtime.submit(
            _proposal(started.operation.id, turn.id)
        )
        await first_runtime.append_observation(
            await domain.project_observation(first_evidence)
        )
        after_first = await first_runtime.inspect(started.operation.id)
        assert after_first.loop_state.observation_characters <= 60

        second_evidence = await first_runtime.submit(
            _proposal(
                started.operation.id,
                turn.id,
                call_id="call-beta",
                key="beta",
            )
        )
        await first_runtime.append_observation(
            await domain.project_observation(second_evidence)
        )
        before_restart = await first_runtime.inspect(started.operation.id)
        assert before_restart.loop_state.observation_characters > 60
        second_task = before_restart.tasks[1]
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        resumed_domain = CountingDomain(resumed_registry)
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider(()),
            context_builder=CountingContextBuilder(),
            domain=resumed_domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "observation_budget_exhausted"
    budget_event = next(
        event for event in final.events if event.type == "budget.exhausted"
    )
    assert budget_event.call_id == "call-beta"
    assert budget_event.task_id == second_task.id
    assert final.observations == before_restart.observations
    assert resumed_domain.validation_calls == 0
    assert resumed_domain.projection_calls == 0
    assert resumed_executor.requests == []


async def test_resume_replays_post_observation_budget_check_without_projection(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "observation-budget.db"
    first_executor = RecordingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _tool_response(),
            budgets=LoopBudgets(max_observation_characters=1),
        )
        evidence = await first_runtime.submit(_proposal(started.operation.id, turn.id))
        observation = await CountingDomain(first_registry).project_observation(evidence)
        await first_runtime.append_observation(observation)
        before_restart = await first_runtime.inspect(started.operation.id)
        assert before_restart.operation.status is OperationStatus.RUNNING
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        domain = CountingDomain(resumed_registry)
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider(()),
            context_builder=CountingContextBuilder(),
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "observation_budget_exhausted"
    assert final.operation.status is OperationStatus.FAILED
    assert final.observations == before_restart.observations
    assert domain.validation_calls == 0
    assert domain.projection_calls == 0
    assert resumed_executor.requests == []


async def test_resume_replays_post_rejection_no_progress_check(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "rejection-check.db"
    executor = RecordingExecutor()
    registry = _registry(executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, turn, _, _ = await _seed_model_call(
            first_runtime,
            registry,
            _tool_response(),
            budgets=LoopBudgets(max_identical_failures=1),
        )
        await first_runtime.record_action_rejection(
            started.operation.id,
            turn.id,
            _tool_response().tool_calls[0],
            ActionRejection(
                code="action.invalid",
                message="The committed action is invalid.",
            ),
        )
    finally:
        await first_store.close()

    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        domain = CountingDomain(registry)
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider(()),
            context_builder=CountingContextBuilder(),
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "no_progress_action_failure_limit"
    assert final.operation.status is OperationStatus.FAILED
    assert domain.validation_calls == 0
    assert executor.requests == []


async def test_resume_started_call_fails_closed_on_provider_identity_change(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "provider-identity.db"
    executor = RecordingExecutor()
    registry = _registry(executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, _, _, _ = await _seed_model_call(first_runtime, registry, None)
    finally:
        await first_store.close()

    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        context = CountingContextBuilder()
        domain = TextDomain(registry)
        provider = MockModelProvider((), provider_id="mock:replacement")
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=context,
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "model_provider_identity_changed"
    assert final.operation.status is OperationStatus.FAILED
    assert provider.requests == ()
    assert context.calls == 0
    assert domain.tool_view_calls == 0


async def test_resume_reuses_committed_denied_readiness_without_reevaluation(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "denied-readiness.db"
    executor = RecordingExecutor()
    registry = _registry(executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, _, _, _ = await _seed_model_call(
            first_runtime,
            registry,
            _final_response(),
        )
        denied = Readiness(
            allowed=False,
            code="ready.missing_fact",
            message="One fact is still missing.",
            evaluated_at=NOW,
            missing_facts=("missing fact",),
        )
        await first_runtime.record_readiness(
            started.operation.id,
            "Recovered answer.",
            denied,
        )
    finally:
        await first_store.close()

    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        context = TextContextBuilder()
        domain = TextDomain(registry)
        provider = MockModelProvider((_final_response(),))
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=context,
            domain=domain,
        )

        result = await loop.resume(started.operation.id)
        final = await resumed_runtime.inspect(started.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert len(final.readiness) == 2
    assert final.readiness[0] == denied
    assert final.readiness[-1].allowed
    assert len(final.turns) == 2
    assert context.calls == 1
    assert domain.readiness_calls == 1
    assert len(provider.requests) == 1


async def test_resume_committed_readiness_and_terminal_redelivery_are_zero_io(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "readiness-terminal.db"
    executor = RecordingExecutor()
    registry = _registry(executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        first_runtime = OperationRuntime(
            capabilities=registry,
            store=first_store,
            clock=lambda: NOW,
        )
        started, _, _, _ = await _seed_model_call(
            first_runtime,
            registry,
            _final_response(),
        )
        await first_runtime.record_readiness(
            started.operation.id,
            "Recovered answer.",
            Readiness(
                allowed=True,
                code="ready.before_restart",
                message="The final candidate is ready.",
                evaluated_at=NOW,
            ),
        )
    finally:
        await first_store.close()

    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        context = CountingContextBuilder()
        domain = TextDomain(registry)
        provider = MockModelProvider(())
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=context,
            domain=domain,
        )

        first_result = await loop.resume(started.operation.id)
        terminal = await resumed_runtime.inspect(started.operation.id)
        terminal_versioned = await resumed_store.load(started.operation.id)
        second_result = await loop.resume(started.operation.id)
        redelivered_result = await loop.run(_trigger())
        changed_budget_loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=context,
            domain=domain,
            budgets=LoopBudgets(max_turns=1),
        )
        changed_budget_result = await changed_budget_loop.run(_trigger())
        mismatched_trigger = AgentTrigger(
            id=_trigger().id,
            agent_id=_trigger().agent_id,
            kind=_trigger().kind,
            source_id=_trigger().source_id,
            payload={"key": "different"},
            created_at=_trigger().created_at,
        )
        with pytest.raises(OperationStateError, match="different operation input"):
            await loop.run(mismatched_trigger)
        after_redelivery = await resumed_runtime.inspect(started.operation.id)
        after_redelivery_versioned = await resumed_store.load(started.operation.id)
    finally:
        await resumed_store.close()

    assert first_result.kind is LoopExitKind.COMPLETED
    assert second_result == first_result
    assert redelivered_result == first_result
    assert changed_budget_result == first_result
    assert after_redelivery == terminal
    assert after_redelivery_versioned == terminal_versioned
    assert context.calls == 0
    assert domain.tool_view_calls == 0
    assert domain.readiness_calls == 0
    assert provider.requests == ()
    assert executor.requests == []


async def test_startup_recovery_uses_ordered_resume_and_continues_after_waiting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    database_path = tmp_path / "startup-recovery.db"
    clock = MutableClock(NOW)
    id_counts: dict[str, int] = {}

    def startup_id(prefix: str) -> str:
        id_counts[prefix] = id_counts.get(prefix, 0) + 1
        return f"{prefix}-startup-{id_counts[prefix]}"

    store = await SQLiteOperationStore.open(database_path, clock=clock)
    executor = RecordingExecutor()
    registry = _registry(executor)
    runtime = OperationRuntime(
        capabilities=registry,
        store=store,
        clock=clock,
        id_factory=startup_id,
    )
    try:
        first = await runtime.begin(
            replace(
                _trigger(),
                id="trigger-startup-a",
                payload={"key": "alpha"},
            )
        )
        clock.current = NOW + timedelta(seconds=1)
        second = await runtime.begin(
            replace(
                _trigger(),
                id="trigger-startup-b",
                payload={"key": "beta"},
            )
        )
        loop = AgentLoop(
            runtime=runtime,
            model=MockModelProvider(()),
            context_builder=CountingContextBuilder(),
            domain=TextDomain(registry),
        )
        resumed_ids: list[str] = []

        async def record_resume(operation_id: str) -> LoopExit:
            resumed_ids.append(operation_id)
            return LoopExit(
                operation_id=operation_id,
                kind=(
                    LoopExitKind.WAITING
                    if operation_id == first.operation.id
                    else LoopExitKind.COMPLETED
                ),
                reason=(
                    "waiting_for_approval"
                    if operation_id == first.operation.id
                    else "ready"
                ),
                final_text=(
                    None if operation_id == first.operation.id else "Recovered."
                ),
                created_at=clock.current,
            )

        async def reject_begin(*args: object, **kwargs: object) -> OperationSnapshot:
            del args, kwargs
            raise AssertionError("startup recovery must not reconstruct a trigger")

        monkeypatch.setattr(loop, "resume", record_resume)
        monkeypatch.setattr(runtime, "begin", reject_begin)

        exits = await loop.recover_startup("agent-restart")
    finally:
        await store.close()

    assert resumed_ids == [first.operation.id, second.operation.id]
    assert tuple(exit.kind for exit in exits) == (
        LoopExitKind.WAITING,
        LoopExitKind.COMPLETED,
    )


async def test_startup_recovery_defers_live_lease_and_drops_new_terminal(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "startup-live-lease.db"
    clock = MutableClock(NOW)
    id_counts: dict[str, int] = {}

    def startup_id(prefix: str) -> str:
        id_counts[prefix] = id_counts.get(prefix, 0) + 1
        return f"{prefix}-startup-live-{id_counts[prefix]}"

    first_executor = ProcessExitExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=clock,
            id_factory=startup_id,
        )
        live, live_turn, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _tool_response(),
        )
        with pytest.raises(AbruptProcessExit):
            await first_runtime.submit(_proposal(live.operation.id, live_turn.id))
        live_before = await first_store.load(live.operation.id)
        assert live_before.snapshot.tasks[0].status is TaskStatus.RUNNING

        clock.current = NOW + timedelta(seconds=1)
        completing_trigger = replace(
            _trigger(),
            id="trigger-startup-completing",
            payload={"key": "completed"},
            created_at=clock.current,
        )
        completing, _, _, _ = await _seed_model_call(
            first_runtime,
            first_registry,
            _final_response(),
            trigger=completing_trigger,
        )
        await first_runtime.record_readiness(
            completing.operation.id,
            "Recovered answer.",
            Readiness(
                allowed=True,
                code="ready.before_startup_recovery",
                message="The committed answer is ready.",
                evaluated_at=clock.current,
            ),
        )
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=clock,
        )
        context = CountingContextBuilder()
        domain = CountingDomain(resumed_registry)
        provider = MockModelProvider(())
        loop = AgentLoop(
            runtime=resumed_runtime,
            model=provider,
            context_builder=context,
            domain=domain,
        )

        first_exits = await loop.recover_startup("agent-restart")
        live_after_first = await resumed_store.load(live.operation.id)
        completed = await resumed_store.load(completing.operation.id)
        second_exits = await loop.recover_startup("agent-restart")
        live_after_second = await resumed_store.load(live.operation.id)
    finally:
        await resumed_store.close()

    assert tuple(exit.operation_id for exit in first_exits) == (
        live.operation.id,
        completing.operation.id,
    )
    assert tuple(exit.kind for exit in first_exits) == (
        LoopExitKind.WAITING,
        LoopExitKind.COMPLETED,
    )
    assert first_exits[0].reason == "task_lease_active"
    assert tuple(exit.operation_id for exit in second_exits) == (live.operation.id,)
    assert second_exits[0].kind is LoopExitKind.WAITING
    assert live_after_first == live_before
    assert live_after_second == live_before
    assert completed.snapshot.operation.status is OperationStatus.SUCCEEDED
    assert context.calls == 0
    assert domain.validation_calls == 0
    assert domain.projection_calls == 0
    assert domain.readiness_calls == 0
    assert provider.requests == ()
    assert resumed_executor.requests == []


@pytest.mark.parametrize(
    "crash_event",
    (
        "turn.created",
        "model_call.started",
        "model_response.recorded",
        "task.created",
        "task.claimed",
        "executor.started",
        "evidence.accepted",
        "observation.recorded",
        "readiness.recorded",
        "operation.succeeded",
    ),
)
async def test_abrupt_exit_after_each_durable_loop_boundary_resumes_exactly(
    tmp_path: Path,
    crash_event: str,
) -> None:
    database_path = tmp_path / f"crash-{crash_event}.db"
    clock = MutableClock(NOW)
    first_executor = RecordingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=clock)
    crash_store = CrashAfterCommittedEventStore(first_store, crash_event)
    first_context = CountingContextBuilder()
    first_domain = CountingDomain(first_registry)
    first_provider = MockModelProvider((_tool_response(), _final_response()))
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=crash_store,
            clock=clock,
        )
        first_loop = AgentLoop(
            runtime=first_runtime,
            model=first_provider,
            context_builder=first_context,
            domain=first_domain,
        )

        with pytest.raises(AbruptProcessExit):
            await first_loop.run(_trigger())

        assert crash_store.crashed
        before = await first_store.load_by_trigger(_trigger().id)
        assert before is not None
        if crash_event == "executor.started":
            clock.current = before.snapshot.task_leases[-1].expires_at + timedelta(
                microseconds=1
            )
    finally:
        await first_store.close()

    remaining_responses = {
        "turn.created": (_tool_response(), _final_response()),
        "model_call.started": (_tool_response(), _final_response()),
        "readiness.recorded": (),
        "operation.succeeded": (),
    }.get(crash_event, (_final_response(),))
    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=clock,
        )
        resumed_context = CountingContextBuilder()
        resumed_domain = CountingDomain(resumed_registry)
        resumed_provider = MockModelProvider(remaining_responses)
        resumed_loop = AgentLoop(
            runtime=resumed_runtime,
            model=resumed_provider,
            context_builder=resumed_context,
            domain=resumed_domain,
        )

        if crash_event == "task.claimed":
            live_before = await resumed_store.load(before.snapshot.operation.id)
            live_exits = await resumed_loop.recover_startup("agent-restart")
            live_after = await resumed_store.load(before.snapshot.operation.id)
            assert tuple(exit.kind for exit in live_exits) == (LoopExitKind.WAITING,)
            assert live_exits[0].reason == "task_lease_active"
            assert live_before == before
            assert live_after == live_before
            assert resumed_context.calls == 0
            assert resumed_domain.validation_calls == 0
            assert resumed_domain.projection_calls == 0
            assert resumed_domain.readiness_calls == 0
            assert resumed_provider.requests == ()
            assert resumed_executor.requests == []
            clock.current = before.snapshot.task_leases[-1].expires_at + timedelta(
                microseconds=1
            )

        if crash_event == "operation.succeeded":
            assert await resumed_runtime.inspect_nonterminal("agent-restart") == ()
            result = await resumed_loop.resume(before.snapshot.operation.id)
        else:
            recovered = await resumed_loop.recover_startup("agent-restart")
            assert tuple(exit.operation_id for exit in recovered) == (
                before.snapshot.operation.id,
            )
            result = recovered[0]
        final = await resumed_store.load(before.snapshot.operation.id)
        redelivered = await resumed_loop.run(_trigger())
        after_redelivery = await resumed_store.load(before.snapshot.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert redelivered == result
    assert after_redelivery == final
    if crash_event == "operation.succeeded":
        assert final == before
    assert final.snapshot.operation.status is OperationStatus.SUCCEEDED
    assert final.snapshot.trigger == before.snapshot.trigger
    assert (
        final.snapshot.events[: len(before.snapshot.events)] == before.snapshot.events
    )
    assert (
        final.snapshot.evidence[: len(before.snapshot.evidence)]
        == before.snapshot.evidence
    )
    assert (
        final.snapshot.observations[: len(before.snapshot.observations)]
        == before.snapshot.observations
    )
    assert (
        final.snapshot.readiness[: len(before.snapshot.readiness)]
        == before.snapshot.readiness
    )
    assert len(final.snapshot.turns) == 2
    assert len(final.snapshot.model_calls) == 2
    assert final.snapshot.loop_state.turn_count == 2
    assert final.snapshot.loop_state.action_count == 1
    assert final.snapshot.loop_state.input_tokens == 12
    assert final.snapshot.loop_state.output_tokens == 5
    assert final.snapshot.loop_state.observation_characters == 51
    assert len(final.snapshot.tasks) == 1
    assert len(final.snapshot.evidence) == 1
    assert len(final.snapshot.observations) == 1
    assert len(final.snapshot.readiness) == 1
    assert final.snapshot.tasks[0].evidence_ids == (final.snapshot.evidence[0].id,)
    assert final.snapshot.observations[0].evidence_id == final.snapshot.evidence[0].id
    followup_request = final.snapshot.model_calls[1].request
    assert [message.role for message in followup_request.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.TOOL,
    ]
    tool_message = followup_request.messages[-1]
    assert len(tool_message.content) == 1
    tool_result = tool_message.content[0]
    assert isinstance(tool_result, ToolResultBlock)
    assert tool_result.call_id == final.snapshot.tasks[0].call_id
    assert tool_result.output == final.snapshot.observations[0].payload
    assert not tool_result.is_error
    if before.snapshot.tasks:
        assert final.snapshot.tasks[0].id == before.snapshot.tasks[0].id
        assert (
            final.snapshot.tasks[0].execution_facts
            == before.snapshot.tasks[0].execution_facts
        )

    expected_attempt = 2 if crash_event in {"task.claimed", "executor.started"} else 1
    assert final.snapshot.tasks[0].attempt == expected_attempt
    assert len(final.snapshot.task_leases) == expected_attempt
    assert tuple(lease.fencing_token for lease in final.snapshot.task_leases) == tuple(
        range(1, expected_attempt + 1)
    )
    expected_release_reasons = {
        "task.claimed": ("expired_before_start", "completed"),
        "executor.started": ("expired_replay_safe", "completed"),
    }.get(crash_event, ("completed",))
    assert (
        tuple(lease.release_reason for lease in final.snapshot.task_leases)
        == expected_release_reasons
    )
    event_types = [event.type for event in final.snapshot.events]
    assert event_types.count("model_call.started") == 2
    assert event_types.count("model_response.recorded") == 2
    assert event_types.count("task.created") == 1
    assert event_types.count("evidence.accepted") == 1
    assert event_types.count("observation.recorded") == 1
    assert event_types.count("readiness.recorded") == 1
    assert event_types.count("operation.succeeded") == 1
    assert event_types.count("task.claimed") == expected_attempt
    assert event_types.count("executor.started") == (
        2 if crash_event == "executor.started" else 1
    )

    expected_initial_provider_calls = {
        "turn.created": 0,
        "model_call.started": 0,
        "readiness.recorded": 2,
        "operation.succeeded": 2,
    }.get(crash_event, 1)
    assert len(first_provider.requests) == expected_initial_provider_calls
    assert len(first_executor.requests) == (
        1
        if crash_event
        in {
            "evidence.accepted",
            "observation.recorded",
            "readiness.recorded",
            "operation.succeeded",
        }
        else 0
    )
    assert len(resumed_provider.requests) == len(remaining_responses)
    assert first_context.calls + resumed_context.calls == 2
    assert first_domain.validation_calls + resumed_domain.validation_calls == 1
    assert first_domain.projection_calls + resumed_domain.projection_calls == 1
    assert first_domain.readiness_calls + resumed_domain.readiness_calls == 1
    assert len(resumed_executor.requests) == (
        1
        if crash_event
        in {
            "turn.created",
            "model_call.started",
            "model_response.recorded",
            "task.created",
            "task.claimed",
            "executor.started",
        }
        else 0
    )
    assert resumed_domain.validation_calls == (
        1
        if crash_event
        in {"turn.created", "model_call.started", "model_response.recorded"}
        else 0
    )
    assert resumed_domain.projection_calls == (
        1
        if crash_event
        in {
            "turn.created",
            "model_call.started",
            "model_response.recorded",
            "task.created",
            "task.claimed",
            "executor.started",
            "evidence.accepted",
        }
        else 0
    )
    assert resumed_domain.readiness_calls == (
        0 if crash_event in {"readiness.recorded", "operation.succeeded"} else 1
    )


async def test_process_exit_inside_safe_executor_reclaims_same_task_after_expiry(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "safe-executor-process-exit.db"
    clock = MutableClock(NOW)
    first_executor = ProcessExitExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=clock,
        )
        first_loop = AgentLoop(
            runtime=first_runtime,
            model=MockModelProvider((_tool_response(),)),
            context_builder=CountingContextBuilder(),
            domain=CountingDomain(first_registry),
        )

        with pytest.raises(AbruptProcessExit):
            await first_loop.run(_trigger())

        before = await first_store.load_by_trigger(_trigger().id)
        assert before is not None
        original_task = before.snapshot.tasks[0]
        assert original_task.status is TaskStatus.RUNNING
        assert original_task.attempt == 1
        assert len(first_executor.requests) == 1
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=clock,
        )
        resumed_loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider((_final_response(),)),
            context_builder=CountingContextBuilder(),
            domain=CountingDomain(resumed_registry),
        )

        live_before = await resumed_store.load(before.snapshot.operation.id)
        live_exits = await resumed_loop.recover_startup("agent-restart")
        live_after = await resumed_store.load(before.snapshot.operation.id)
        assert tuple(exit.operation_id for exit in live_exits) == (
            before.snapshot.operation.id,
        )
        assert live_exits[0].kind is LoopExitKind.WAITING
        assert live_exits[0].reason == "task_lease_active"
        assert live_before == before
        assert live_after == live_before

        clock.current = before.snapshot.task_leases[0].expires_at + timedelta(
            microseconds=1
        )
        recovered = await resumed_loop.recover_startup("agent-restart")
        assert tuple(exit.operation_id for exit in recovered) == (
            before.snapshot.operation.id,
        )
        result = recovered[0]
        final = await resumed_store.load(before.snapshot.operation.id)
    finally:
        await resumed_store.close()

    recovered_task = final.snapshot.tasks[0]
    assert result.kind is LoopExitKind.COMPLETED
    assert recovered_task.id == original_task.id
    assert recovered_task.execution_facts == original_task.execution_facts
    assert recovered_task.status is TaskStatus.SUCCEEDED
    assert recovered_task.attempt == 2
    assert tuple(lease.fencing_token for lease in final.snapshot.task_leases) == (1, 2)
    assert tuple(lease.release_reason for lease in final.snapshot.task_leases) == (
        "expired_replay_safe",
        "completed",
    )
    assert len(resumed_executor.requests) == 1
    assert resumed_executor.requests[0].task_id == original_task.id
    assert resumed_executor.requests[0].attempt == 2
    assert resumed_executor.requests[0].fencing_token == 2
    assert final.snapshot.loop_state.action_count == 1
    assert len(final.snapshot.tasks) == 1
    assert len(final.snapshot.evidence) == 1
    assert final.snapshot.evidence[0].attempt == 2
    assert len(final.snapshot.observations) == 1
    event_types = [event.type for event in final.snapshot.events]
    assert event_types.count("task.created") == 1
    assert event_types.count("task.claimed") == 2
    assert event_types.count("executor.started") == 2
    assert event_types.count("evidence.accepted") == 1


async def test_provider_result_lost_before_response_commit_resends_exact_request(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "unknown-model-outcome.db"
    first_executor = RecordingExecutor()
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    first_provider = MockModelProvider((_tool_response(),))
    try:
        first_runtime = CrashBeforeModelResponseRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )
        first_loop = AgentLoop(
            runtime=first_runtime,
            model=first_provider,
            context_builder=CountingContextBuilder(),
            domain=CountingDomain(first_registry),
        )

        with pytest.raises(AbruptProcessExit):
            await first_loop.run(_trigger())

        before = await first_store.load_by_trigger(_trigger().id)
        assert before is not None
        started_call = before.snapshot.model_calls[0]
        assert started_call.response is None
        assert first_provider.requests == (started_call.request,)
    finally:
        await first_store.close()

    resumed_executor = RecordingExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=lambda: NOW)
    resumed_provider = MockModelProvider((_tool_response(), _final_response()))
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        resumed_loop = AgentLoop(
            runtime=resumed_runtime,
            model=resumed_provider,
            context_builder=CountingContextBuilder(),
            domain=CountingDomain(resumed_registry),
        )

        recovered = await resumed_loop.recover_startup("agent-restart")
        assert tuple(exit.operation_id for exit in recovered) == (
            before.snapshot.operation.id,
        )
        result = recovered[0]
        final = await resumed_store.load(before.snapshot.operation.id)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert resumed_provider.requests[0] == started_call.request
    assert final.snapshot.model_calls[0].id == started_call.id
    assert final.snapshot.model_calls[0].response == _tool_response()
    assert final.snapshot.loop_state.input_tokens == 12
    assert final.snapshot.loop_state.output_tokens == 5
    first_call_events = [
        event.type
        for event in final.snapshot.events
        if event.model_call_id == started_call.id
    ]
    assert first_call_events.count("model_call.started") == 1
    assert first_call_events.count("model_response.recorded") == 1
    assert first_executor.requests == []
    assert len(resumed_executor.requests) == 1


async def test_blob_put_crash_leaves_unlinked_orphan_then_safe_retry_links_new_blob(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "blob-orphan.db"
    blob_root = tmp_path / "blobs"
    clock = MutableClock(NOW)
    first_executor = ArtifactExecutor()
    first_registry = _registry(first_executor)
    crashing_blobs = CrashAfterPutBlobStore(blob_root)
    first_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        first_runtime = OperationRuntime(
            capabilities=first_registry,
            store=first_store,
            blob_store=crashing_blobs,
            clock=clock,
            id_factory=NamespacedIds("before-crash"),
        )
        first_loop = AgentLoop(
            runtime=first_runtime,
            model=MockModelProvider((_tool_response(),)),
            context_builder=CountingContextBuilder(),
            domain=CountingDomain(first_registry),
        )

        with pytest.raises(AbruptProcessExit):
            await first_loop.run(_trigger())

        before = await first_store.load_by_trigger(_trigger().id)
        assert before is not None
        assert crashing_blobs.committed is not None
        orphan = crashing_blobs.committed
        assert before.snapshot.tasks[0].status is TaskStatus.RUNNING
        assert before.snapshot.evidence == ()
        clock.current = before.snapshot.task_leases[-1].expires_at + timedelta(
            microseconds=1
        )
    finally:
        await first_store.close()

    resumed_executor = ArtifactExecutor()
    resumed_registry = _registry(resumed_executor)
    resumed_blobs = LocalBlobStore(blob_root)
    resumed_store = await SQLiteOperationStore.open(database_path, clock=clock)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            blob_store=resumed_blobs,
            clock=clock,
            id_factory=NamespacedIds("after-crash"),
        )
        resumed_loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider((_final_response(),)),
            context_builder=CountingContextBuilder(),
            domain=CountingDomain(resumed_registry),
        )

        recovered = await resumed_loop.recover_startup("agent-restart")
        assert tuple(exit.operation_id for exit in recovered) == (
            before.snapshot.operation.id,
        )
        result = recovered[0]
        final = await resumed_store.load(before.snapshot.operation.id)
        orphan_after = await resumed_blobs.metadata(orphan.blob_id)
        orphan_reader = await resumed_blobs.open(orphan.blob_id)
        async with orphan_reader:
            orphan_content = await orphan_reader.read(orphan.size_bytes)
    finally:
        await resumed_store.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert orphan_after == orphan
    assert orphan_content == b'{"key":"alpha","value":"ALPHA"}'
    assert orphan.blob_id.startswith("blob-before-crash-")
    assert orphan.evidence_id is not None
    assert orphan.evidence_id.startswith("evidence-before-crash-")
    assert final.snapshot.tasks[0].attempt == 2
    assert tuple(lease.release_reason for lease in final.snapshot.task_leases) == (
        "expired_replay_safe",
        "completed",
    )
    assert len(final.snapshot.evidence) == 1
    assert final.snapshot.evidence[0].attempt == 2
    assert final.snapshot.evidence[0].blob_id is not None
    assert final.snapshot.evidence[0].blob_id.startswith("blob-after-crash-")
    assert final.snapshot.evidence[0].id.startswith("evidence-after-crash-")
    assert final.snapshot.evidence[0].blob_id != orphan.blob_id
    assert final.snapshot.evidence[0].id != orphan.evidence_id
    assert all(item.blob_id != orphan.blob_id for item in final.snapshot.evidence)
    assert all(
        event.evidence_id != orphan.evidence_id for event in final.snapshot.events
    )
    assert len(first_executor.requests) == 1
    assert len(resumed_executor.requests) == 1
    assert resumed_executor.requests[0].attempt == 2
    assert resumed_executor.requests[0].fencing_token == 2
