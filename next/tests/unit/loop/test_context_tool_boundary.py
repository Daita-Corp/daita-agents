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
from daita.context import RequiredContextOverflow
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

    async def tool_views(
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


class OverflowingContextBuilder:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        del operation, turn, tools
        raise RequiredContextOverflow(
            profile_id="mock:context-overflow",
            input_limit_tokens=100,
            output_reserve_tokens=20,
            tool_tokens=10,
            required_system_tokens=10,
            required_routing_tokens=5,
            required_intent_tokens=5,
            current_operation_envelope_tokens=20,
            current_operation_body_tokens=10,
            minimum_session_tokens=20,
            projected_session_tokens=30,
            required_tokens=70,
            available_tokens=60,
            optional_omitted_tokens=40,
        )


class SequentialPolicyContextBuilder:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Require sequential tool calls."),),
                ),
            ),
            tools=tools,
            allow_parallel_tool_calls=False,
        )


class ExactProjectionContextBuilder:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert tuple(tool.name for tool in tools) == ("tool_a",)
        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Use only the tool exposed for this call."),),
                ),
            ),
            tools=tools,
        )


class HiddenArbitraryExtensionDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry
        self.validated_call_ids: list[str] = []

    async def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        assert operation.operation.status is OperationStatus.RUNNING
        return (self.registry.tool_definition("tool_a"),)

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        self.validated_call_ids.append(call.id)
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
        raise AssertionError("an unprojected extension must not produce evidence")

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        assert text == "The unavailable extension was not executed."
        assert not operation.tasks
        assert not operation.evidence
        assert operation.observations[-1].code == "action.tool_not_projected"
        return Readiness(
            allowed=True,
            code="ready.hidden_extension_rejected",
            message="The unavailable extension call was rejected.",
            evaluated_at=NOW,
        )


class UnsupportedPolicyProvider:
    provider_id = "mock:unsupported-policy"

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def supports_request_policy(self, request: ModelRequest) -> bool:
        return False

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        raise AssertionError("unsupported request policy must fail before provider I/O")


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


async def test_provider_emitted_arbitrary_extension_absent_from_exact_request_fails_closed() -> (
    None
):
    executor_a = NeverCalledExecutor("fake.a.executor")
    extension_executor = NeverCalledExecutor("fake.warehouse.executor")
    registry = CapabilityRegistry(
        capabilities=(_capability("a"), _capability("warehouse")),
        executors=(executor_a, extension_executor),
        tool_views=(
            ToolView(
                name="tool_a",
                capability_id="fake.a",
                description="Read registered fake value A.",
            ),
            ToolView(
                name="warehouse_lookup",
                capability_id="fake.warehouse",
                description="Look up a value through an arbitrary extension.",
            ),
        ),
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            ModelResponse(
                finish_reason=FinishReason.TOOL_CALLS,
                tool_calls=(
                    ToolCall(
                        id="hidden-extension-call",
                        name="warehouse_lookup",
                        arguments={"key": "alpha"},
                    ),
                ),
            ),
            ModelResponse(
                text="The unavailable extension was not executed.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    domain = HiddenArbitraryExtensionDomain(registry)
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=ExactProjectionContextBuilder(),
        domain=domain,
    )

    result = await loop.run(
        AgentTrigger(
            id="trigger-hidden-arbitrary-extension",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "Use only tool A."},
            created_at=NOW,
        )
    )
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.COMPLETED
    assert domain.validated_call_ids == ["hidden-extension-call"]
    assert all(
        tuple(tool.name for tool in request.tools) == ("tool_a",)
        for request in provider.requests
    )
    assert final.tasks == ()
    assert final.evidence == ()
    assert len(final.observations) == 1
    rejection = final.observations[0]
    assert rejection.call_id == "hidden-extension-call"
    assert rejection.code == "action.tool_not_projected"
    assert dict(rejection.payload) == {
        "capability_id": "fake.warehouse",
        "tool_name": "warehouse_lookup",
    }
    assert executor_a.requests == []
    assert extension_executor.requests == []


async def test_unsupported_request_policy_is_persisted_before_provider_io() -> None:
    executor = NeverCalledExecutor("fake.a.executor")
    registry = CapabilityRegistry(
        capabilities=(_capability("a"),),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="tool_a",
                capability_id="fake.a",
                description="Read registered fake value A.",
            ),
        ),
    )
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = UnsupportedPolicyProvider()
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=SequentialPolicyContextBuilder(),
        domain=RestrictedDomain(registry),
    )

    result = await loop.run(
        AgentTrigger(
            id="trigger-unsupported-policy",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "Require sequential tool calls."},
            created_at=NOW,
        )
    )
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "invalid_request"
    assert final.operation.terminal_reason == "invalid_request"
    assert final.model_calls[-1].request.allow_parallel_tool_calls is False
    assert final.model_calls[-1].error_code == "invalid_request"
    assert provider.requests == []
    assert executor.requests == []


async def test_required_context_overflow_is_typed_and_never_reaches_provider() -> None:
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
                text="The provider must never receive an overflowed request.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=OverflowingContextBuilder(),
        domain=RestrictedDomain(registry),
    )

    result = await loop.run(
        AgentTrigger(
            id="trigger-context-overflow",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "secret sentinel must never be persisted"},
            created_at=NOW,
        )
    )
    final = await runtime.inspect(result.operation_id)

    assert result.kind is LoopExitKind.FAILED
    assert result.reason == "context.required_overflow"
    assert final.operation.terminal_reason == "context.required_overflow"
    assert final.model_calls == ()
    assert provider.requests == ()
    assert executor_a.requests == []
    assert executor_b.requests == []
    assert [event.type for event in final.events] == [
        "trigger.received",
        "operation.created",
        "turn.created",
        "context.required_overflow",
        "operation.failed",
    ]
    overflow = final.events[-2]
    assert overflow.payload["code"] == "context.required_overflow"
    assert overflow.payload["total_required_tokens"] == 80
    assert "secret sentinel" not in repr(overflow.payload)
