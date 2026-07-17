from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone

import pytest

from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityExecutionError,
    CapabilityRegistry,
    EvidenceCandidate,
    EvidenceValidationError,
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
from daita.loop.models import LoopPhase, Readiness
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Observation,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime, OperationStateError

NOW = datetime(2026, 7, 17, 9, 0, tzinfo=timezone.utc)


class CandidateExecutor:
    def __init__(self, executor_id: str, candidate: EvidenceCandidate) -> None:
        self.executor_id = executor_id
        self.candidate = candidate
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        return self.candidate


class IdentityMutatingExecutor(CandidateExecutor):
    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        self.executor_id = "mutated.executor"
        return self.candidate


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
) -> tuple[OperationRuntime, CandidateExecutor, str, str]:
    executor = executor_override or CandidateExecutor("fake.read.executor", candidate)
    registry = _registry(executor, include_other=include_other)
    if id_factory is None:
        runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    else:
        runtime = OperationRuntime(
            capabilities=registry,
            clock=lambda: NOW,
            id_factory=id_factory,
        )
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-1",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "read alpha"},
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


async def test_executor_start_failure_retains_the_committed_pending_task() -> None:
    counter = 0
    task_id_allocated = False
    events_after_task = 0

    def fail_on_executor_started_event(prefix: str) -> str:
        nonlocal counter, task_id_allocated, events_after_task
        counter += 1
        if prefix == "task":
            task_id_allocated = True
        elif task_id_allocated and prefix == "event":
            events_after_task += 1
            if events_after_task == 2:
                raise RuntimeError("injected executor.started commit failure")
        return f"{prefix}-{counter}"

    candidate = EvidenceCandidate(
        kind="fake.read.result",
        schema_version=1,
        payload={"key": "alpha", "value": "ALPHA"},
    )
    runtime, executor, operation_id, turn_id = await _runtime_with_committed_tool_call(
        candidate,
        id_factory=fail_on_executor_started_event,
    )

    with pytest.raises(RuntimeError, match="executor.started commit failure"):
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
