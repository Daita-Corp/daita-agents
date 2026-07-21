from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path

import pytest

from daita import Agent
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
    MessageRole,
    ModelRequest,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from daita.llm.providers.openai import OpenAIResponsesProvider
from daita.loop.models import LoopBudgets, LoopExitKind, Readiness, Turn
from daita.operations.checkpoints import ModelCallStatus, OperationSnapshot
from daita.operations.models import (
    ActionProposal,
    ActionRejection,
    Evidence,
    Observation,
    OperationStatus,
    TaskStatus,
)

FAKE_KEY = "persistence-probe"
FAKE_VALUE = "PERSISTED-OPENAI-EVIDENCE"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _fake_read_capability() -> Capability:
    return Capability(
        id="acceptance.fake.read",
        owner="openai-live-acceptance",
        description="Read one test-owned deterministic value.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="acceptance.fake.read.result",
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
        executor_id="acceptance.fake.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


class LiveFakeReadExecutor:
    executor_id = "acceptance.fake.read.executor"

    def __init__(self) -> None:
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        assert request.capability_id == "acceptance.fake.read"
        assert request.arguments["key"] == FAKE_KEY
        self.requests.append(request)
        return EvidenceCandidate(
            kind="acceptance.fake.read.result",
            schema_version=1,
            payload={"key": FAKE_KEY, "value": FAKE_VALUE},
        )


class LiveFakeReadDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    async def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return self._registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal | ActionRejection:
        if operation.tasks:
            return ActionRejection(
                code="fake_read_already_completed",
                message="The acceptance fake value may be read only once.",
            )
        try:
            view, capability = self._registry.resolve_tool(call.name)
            arguments = self._registry.validate_arguments(
                capability.id,
                call.arguments,
            )
        except (KeyError, ValueError) as error:
            return ActionRejection(
                code="invalid_fake_read",
                message="Call read_fake_value with the required string key.",
                details={"error": str(error)},
            )
        if arguments["key"] != FAKE_KEY:
            return ActionRejection(
                code="invalid_fake_key",
                message=f"The only accepted key is {FAKE_KEY}.",
            )
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=view.capability_id,
            arguments=arguments,
            proposed_at=_now(),
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        assert evidence.accepted
        return Observation(
            operation_id=evidence.operation_id,
            turn_id=evidence.turn_id,
            code="acceptance.fake.read.succeeded",
            message="The test-owned fake value was read.",
            payload=evidence.payload,
            success=True,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            created_at=_now(),
        )

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        accepted = tuple(item for item in operation.evidence if item.accepted)
        observed_ids = {
            observation.evidence_id
            for observation in operation.observations
            if observation.success
        }
        has_durable_value = (
            len(accepted) == 1
            and accepted[0].id in observed_ids
            and accepted[0].payload.get("value") == FAKE_VALUE
        )
        reports_value = FAKE_VALUE.casefold() in text.casefold()
        missing: list[str] = []
        if not has_durable_value:
            missing.append("Call read_fake_value once before answering.")
        if not reports_value:
            missing.append(f"Report the exact value {FAKE_VALUE}.")
        return Readiness(
            allowed=not missing,
            code="ready.acceptance_fake_read" if not missing else "missing_fake_read",
            message=(
                "The persisted fake read is reflected in the final answer."
                if not missing
                else "The answer must be grounded in the persisted fake read."
            ),
            missing_facts=tuple(missing),
            evaluated_at=_now(),
        )


class LiveFakeReadContext:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        assert [tool.name for tool in tools] == ["read_fake_value"]
        initial_turn_id = operation.turns[0].id
        message = operation.trigger.payload["message"]
        assert isinstance(message, str)
        messages: list[CanonicalMessage] = [
            CanonicalMessage(
                agent_id=operation.operation.agent_id,
                operation_id=operation.operation.id,
                session_id=operation.operation.session_id,
                turn_id=initial_turn_id,
                role=MessageRole.SYSTEM,
                content=(
                    TextBlock(
                        "This is a persistence acceptance check. You must call "
                        f"read_fake_value exactly once with key {FAKE_KEY!r}. After "
                        "the tool result, answer briefly and include its exact value."
                    ),
                ),
            ),
            CanonicalMessage(
                agent_id=operation.operation.agent_id,
                operation_id=operation.operation.id,
                session_id=operation.operation.session_id,
                turn_id=initial_turn_id,
                role=MessageRole.USER,
                content=(TextBlock(message),),
            ),
        ]
        task_by_call_id = {task.call_id: task for task in operation.tasks}
        observation_by_task_id = {
            observation.task_id: observation
            for observation in operation.observations
            if observation.task_id is not None
        }

        for model_call in operation.model_calls:
            if (
                model_call.status is not ModelCallStatus.COMPLETED
                or model_call.response is None
            ):
                continue
            response = model_call.response
            content = () if response.text is None else (TextBlock(response.text),)
            messages.append(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=model_call.turn_id,
                    role=MessageRole.ASSISTANT,
                    content=content,
                    tool_calls=response.tool_calls,
                    provider_metadata=response.provider_metadata,
                )
            )
            for call in response.tool_calls:
                task = task_by_call_id.get(call.id)
                if task is None:
                    continue
                observation = observation_by_task_id.get(task.id)
                if observation is None:
                    continue
                messages.append(
                    CanonicalMessage(
                        agent_id=operation.operation.agent_id,
                        operation_id=operation.operation.id,
                        session_id=operation.operation.session_id,
                        turn_id=model_call.turn_id,
                        role=MessageRole.TOOL,
                        content=(
                            ToolResultBlock(
                                call_id=call.id,
                                output=observation.payload,
                                is_error=not observation.success,
                            ),
                        ),
                    )
                )

        if operation.readiness and not operation.readiness[-1].allowed:
            messages.append(
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    session_id=operation.operation.session_id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(
                        TextBlock(
                            "Correct the prior response: call the tool if needed, then "
                            f"include the exact observed value {FAKE_VALUE}."
                        ),
                    ),
                )
            )

        return ModelRequest(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            messages=tuple(messages),
            tools=tools,
        )


@pytest.mark.requires_llm
async def test_live_openai_tool_loop_survives_public_agent_reopen(
    tmp_path: Path,
) -> None:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model_name = os.getenv("DAITA_OPENAI_MODEL", "").strip()
    if not api_key or not model_name:
        pytest.skip("requires OPENAI_API_KEY and an explicit DAITA_OPENAI_MODEL")

    executor = LiveFakeReadExecutor()
    capability = _fake_read_capability()
    registry = CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_fake_value",
                capability_id=capability.id,
                description=(
                    "Read the one deterministic persistence-probe value. Call once."
                ),
            ),
        ),
    )
    provider = OpenAIResponsesProvider(
        model=model_name,
        api_key=api_key,
    )
    agent = await Agent.create(
        "openai-live-persistence",
        root=tmp_path,
        model=provider,
        context_builder=LiveFakeReadContext(),
        domain=LiveFakeReadDomain(registry),
        capabilities=registry,
        budgets=LoopBudgets(
            max_turns=5,
            max_actions=2,
            max_repairs=2,
            max_total_tokens=20_000,
            max_wall_time_seconds=180,
            task_timeout_seconds=30,
        ),
    )
    try:
        result = await agent.run(
            f"Read {FAKE_KEY!r} with the provided tool and report its exact value.",
            session_id="live-openai-session",
        )
    finally:
        await agent.close()

    assert result.kind is LoopExitKind.COMPLETED
    assert result.final_text is not None
    assert FAKE_VALUE.casefold() in result.final_text.casefold()
    assert len(executor.requests) == 1

    reopened = await Agent.open(
        "openai-live-persistence",
        root=tmp_path,
    )
    try:
        persisted = await reopened.inspect(result.operation_id)
    finally:
        await reopened.close()

    assert persisted.operation.status is OperationStatus.SUCCEEDED
    assert persisted.operation.final_text == result.final_text
    assert persisted.operation.terminal_reason == result.reason
    assert persisted.loop_state.final_answer_candidate == result.final_text

    assert len(persisted.model_calls) >= 2
    assert all(
        call.status is ModelCallStatus.COMPLETED for call in persisted.model_calls
    )
    assert all(call.response is not None for call in persisted.model_calls)
    tool_responses = tuple(
        call.response
        for call in persisted.model_calls
        if call.response is not None and call.response.tool_calls
    )
    assert len(tool_responses) == 1
    assert len(tool_responses[0].tool_calls) == 1
    persisted_tool_call = tool_responses[0].tool_calls[0]
    assert persisted_tool_call.provider_call_id is not None

    assert len(persisted.tasks) == 1
    task = persisted.tasks[0]
    assert task.call_id == persisted_tool_call.id
    assert task.status is TaskStatus.SUCCEEDED
    assert task.capability_id == capability.id
    assert task.arguments["key"] == FAKE_KEY

    assert len(persisted.evidence) == 1
    evidence = persisted.evidence[0]
    assert evidence.accepted
    assert evidence.task_id == task.id
    assert evidence.id in task.evidence_ids
    assert evidence.payload["key"] == FAKE_KEY
    assert evidence.payload["value"] == FAKE_VALUE

    assert len(persisted.observations) == 1
    observation = persisted.observations[0]
    assert observation.success
    assert observation.task_id == task.id
    assert observation.evidence_id == evidence.id
    assert observation.payload == evidence.payload

    event_types = tuple(event.type for event in persisted.events)
    assert event_types.count("model_call.started") == len(persisted.model_calls)
    assert event_types.count("model_response.recorded") == len(persisted.model_calls)
    assert event_types.count("task.created") == 1
    assert event_types.count("evidence.accepted") == 1
    assert event_types.count("observation.recorded") == 1
    assert event_types[-1] == "operation.succeeded"
