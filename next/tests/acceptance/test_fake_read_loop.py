from __future__ import annotations

from datetime import datetime, timezone

import pytest

from daita._json import FrozenJsonObject
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
from daita.loop.models import LoopExitKind, Readiness, Turn
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime, OperationSnapshot

NOW = datetime(2026, 7, 16, 14, 0, tzinfo=timezone.utc)


def _capability() -> Capability:
    return Capability(
        id="fake.read",
        owner="loop-lab",
        description="Read one deterministic fake value.",
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


class RecordingReadExecutor:
    executor_id = "fake.read.executor"

    def __init__(self) -> None:
        self.runtime: OperationRuntime | None = None
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        assert self.runtime is not None
        at_entry = await self.runtime.inspect(request.operation_id)
        task = next(task for task in at_entry.tasks if task.id == request.task_id)
        assert task.status is TaskStatus.RUNNING
        task_events = [
            event.type for event in at_entry.events if event.task_id == request.task_id
        ]
        assert task_events == [
            "task.created",
            "task.ready",
            "task.claimed",
            "executor.started",
        ]
        lease = next(
            lease
            for lease in at_entry.task_leases
            if lease.task_id == request.task_id and lease.released_at is None
        )
        assert lease.started_at is not None
        assert request.attempt == task.attempt == lease.attempt
        assert request.fencing_token == lease.fencing_token
        assert request.executor_id == task.executor_id
        assert request.idempotency_key == task.execution_facts.idempotency_key
        assert not any(
            evidence.task_id == request.task_id for evidence in at_entry.evidence
        )
        self.requests.append(request)
        key = request.arguments["key"]
        assert isinstance(key, str)
        return EvidenceCandidate(
            kind="fake.read.result",
            schema_version=1,
            payload={"key": key, "value": key.upper()},
        )


class FakeReadDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

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
        view, capability = self.registry.resolve_tool(call.name)
        arguments = self.registry.validate_arguments(capability.id, call.arguments)
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=view.capability_id,
            arguments=arguments,
            proposed_at=NOW,
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        assert evidence.accepted
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
        assert text == "The requested fake values were read."
        assert operation.evidence
        assert all(evidence.accepted for evidence in operation.evidence)
        assert len(operation.observations) == len(operation.evidence)
        return Readiness(
            allowed=True,
            code="ready.fake_read",
            message="Every requested fake value has current accepted evidence.",
            evaluated_at=NOW,
        )


class TranscriptContextBuilder:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert [tool.name for tool in tools] == ["read_fake_value"]
        messages: tuple[CanonicalMessage, ...]
        if not operation.model_calls:
            messages = (
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Read the requested fake values."),),
                ),
            )
        else:
            first_call = operation.model_calls[0]
            assert first_call.response is not None
            messages_list = list(first_call.request.messages)
            messages_list.append(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=first_call.turn_id,
                    role=MessageRole.ASSISTANT,
                    tool_calls=first_call.response.tool_calls,
                )
            )
            for observation in operation.observations:
                assert observation.task_id is not None
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
                                is_error=not observation.success,
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


@pytest.mark.parametrize("keys", [("alpha",), ("alpha", "beta")])
async def test_fake_reads_follow_the_only_durable_executor_path_in_order(
    keys: tuple[str, ...],
) -> None:
    executor = RecordingReadExecutor()
    capability = _capability()
    registry = CapabilityRegistry(
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
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    executor.runtime = runtime
    tool_calls = tuple(
        ToolCall(
            id=f"call-{index}",
            name="read_fake_value",
            arguments={"key": key},
        )
        for index, key in enumerate(keys, start=1)
    )
    provider = MockModelProvider(
        (
            ModelResponse(
                tool_calls=tool_calls,
                finish_reason=FinishReason.TOOL_CALLS,
                usage=ModelUsage(input_tokens=10, output_tokens=4),
            ),
            ModelResponse(
                text="The requested fake values were read.",
                finish_reason=FinishReason.STOP,
                usage=ModelUsage(input_tokens=20, output_tokens=7),
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=TranscriptContextBuilder(),
        domain=FakeReadDomain(registry),
    )
    trigger = AgentTrigger(
        id=f"trigger-{'-'.join(keys)}",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        payload={"keys": list(keys)},
        created_at=NOW,
    )

    result = await loop.run(trigger)
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert final.operation.status is OperationStatus.SUCCEEDED
    assert [request.arguments["key"] for request in executor.requests] == list(keys)
    assert len(final.tasks) == len(keys)
    assert all(task.status is TaskStatus.SUCCEEDED for task in final.tasks)
    assert all(len(task.evidence_ids) == 1 for task in final.tasks)
    assert len(final.evidence) == len(keys)
    assert all(evidence.accepted for evidence in final.evidence)
    assert all(
        evidence.content_hash.startswith("sha256:") for evidence in final.evidence
    )
    assert len(final.observations) == len(keys)
    assert final.loop_state.turn_count == 2
    assert final.loop_state.action_count == len(keys)
    assert final.loop_state.input_tokens == 30
    assert final.loop_state.output_tokens == 11

    action_events = [
        "task.created",
        "task.ready",
        "task.claimed",
        "executor.started",
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
        "observation.recorded",
    ]
    assert [event.type for event in final.events] == [
        "trigger.received",
        "operation.created",
        "turn.created",
        "context.built",
        "model_call.started",
        "model_response.recorded",
        *(event for _ in keys for event in action_events),
        "turn.created",
        "context.built",
        "model_call.started",
        "model_response.recorded",
        "readiness.recorded",
        "operation.succeeded",
    ]
    model_call_by_turn = {
        model_call.turn_id: model_call.id for model_call in final.model_calls
    }
    correlated_event_types = {
        "context.built",
        "model_call.started",
        "model_response.recorded",
        "task.created",
        "task.ready",
        "task.claimed",
        "executor.started",
        "executor.completed",
        "evidence.accepted",
        "task.succeeded",
        "observation.recorded",
        "readiness.recorded",
    }
    for event in final.events:
        if event.type not in correlated_event_types:
            continue
        assert event.turn_id is not None
        assert event.model_call_id == model_call_by_turn[event.turn_id]
    second_request = provider.requests[1]
    assert [message.role for message in second_request.messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        *(MessageRole.TOOL for _ in keys),
    ]
    for message in second_request.messages[2:]:
        result_block = message.content[0]
        assert isinstance(result_block, ToolResultBlock)
        assert isinstance(result_block.output, FrozenJsonObject)
    provider.assert_consumed()
