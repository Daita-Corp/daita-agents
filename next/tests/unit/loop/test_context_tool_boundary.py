from __future__ import annotations

from datetime import datetime, timezone

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
    TextBlock,
    ToolCall,
    ToolDefinition,
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
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime, OperationSnapshot

NOW = datetime(2026, 7, 17, 10, 0, tzinfo=timezone.utc)


class NeverCalledExecutor:
    def __init__(self, executor_id: str) -> None:
        self.executor_id = executor_id
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        raise AssertionError("context tool expansion must not reach an executor")


def _capability(name: str) -> Capability:
    return Capability(
        id=f"fake.{name}",
        owner="loop-lab",
        description=f"Read fake value {name}.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind=f"fake.{name}.result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        executor_id=f"fake.{name}.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


class RestrictedDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        assert operation.operation.status is OperationStatus.RUNNING
        return (self.registry.tool_definitions()[0],)

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        raise AssertionError("model action processing must not begin")

    async def project_observation(self, evidence: Evidence) -> Observation:
        raise AssertionError("observation projection must not begin")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        raise AssertionError("readiness evaluation must not begin")


class ExpandingContextBuilder:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert [tool.name for tool in tools] == ["tool_a"]
        assert [tool.name for tool in self.registry.tool_definitions()] == [
            "tool_a",
            "tool_b",
        ]
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Attempt to expand the tool set."),),
                ),
            ),
            tools=self.registry.tool_definitions(),
        )


async def test_context_builder_cannot_expand_domain_tool_projection() -> None:
    executor_a = NeverCalledExecutor("fake.a.executor")
    executor_b = NeverCalledExecutor("fake.b.executor")
    registry = CapabilityRegistry(
        capabilities=(_capability("a"), _capability("b")),
        executors=(executor_a, executor_b),
        tool_views=(
            ToolView(
                name="tool_a",
                capability_id="fake.a",
                description="Read registered fake value A.",
            ),
            ToolView(
                name="tool_b",
                capability_id="fake.b",
                description="Read registered fake value B.",
            ),
        ),
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            ModelResponse(
                text="The provider must never see the expanded request.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=ExpandingContextBuilder(registry),
        domain=RestrictedDomain(registry),
    )

    result = await loop.run(
        AgentTrigger(
            id="trigger-tool-expansion",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "Use only tool A."},
            created_at=NOW,
        )
    )
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "context_build_failed"
    assert final.operation.status is OperationStatus.FAILED
    assert final.turns[-1].model_request_id is None
    assert final.model_calls == ()
    assert final.tasks == ()
    assert final.evidence == ()
    assert final.observations == ()
    assert provider.requests == ()
    assert executor_a.requests == []
    assert executor_b.requests == []
    assert [event.type for event in final.events] == [
        "trigger.received",
        "operation.created",
        "turn.created",
        "operation.failed",
    ]
