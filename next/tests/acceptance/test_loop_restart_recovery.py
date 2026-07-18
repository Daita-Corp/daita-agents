from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityRegistry,
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
from daita.loop.models import LoopExitKind, LoopPhase, Readiness, Turn
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime
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


def _registry(executor: RecordingExecutor) -> CapabilityRegistry:
    capability = _capability()
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
        assert len(operation.evidence) == len(operation.observations) == 1
        return Readiness(
            allowed=True,
            code="ready.recovered",
            message="Recovered evidence is ready.",
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
            task = operation.tasks[0]
            observation = operation.observations[0]
            messages = (
                *first_call.request.messages,
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=first_call.turn_id,
                    role=MessageRole.ASSISTANT,
                    tool_calls=first_call.response.tool_calls,
                ),
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
                ),
            )
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=messages,
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
