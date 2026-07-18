from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
import sqlite3

import pytest

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
    ToolResultBlock,
)
from daita.llm.providers.mock import MockModelProvider
from daita.loop.driver import AgentLoop
from daita.loop.models import LoopExitKind, LoopPhase, Readiness, Turn
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.governance import (
    ApprovalStatus,
    DefaultPolicyEvaluator,
    DefaultPolicyProfile,
    GovernanceDecision,
    GovernanceFacts,
    PolicyEffect,
)
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime, OperationStateError
from daita.operations.store import InMemoryOperationStore
from daita.storage.sqlite import SQLiteOperationStore

NOW = datetime(2026, 7, 18, 16, 0, tzinfo=timezone.utc)


class DurableMarkerExecutor:
    """Test-owned idempotent side effect; production code never owns the marker."""

    executor_id = "test.marker.executor"

    def __init__(self, path: Path) -> None:
        self.path = path
        self.requests: list[ExecutionRequest] = []
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS marker ("
                "idempotency_key TEXT PRIMARY KEY, value TEXT NOT NULL)"
            )

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        assert request.idempotency_key is not None
        value = request.arguments["value"]
        assert isinstance(value, str)
        self.requests.append(request)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO marker(idempotency_key, value) VALUES (?, ?)",
                (request.idempotency_key, value),
            )
            row = connection.execute(
                "SELECT value FROM marker WHERE idempotency_key = ?",
                (request.idempotency_key,),
            ).fetchone()
        assert row is not None
        return EvidenceCandidate(
            kind="test.marker.result",
            schema_version=1,
            payload={"value": row[0]},
        )

    def values(self) -> tuple[str, ...]:
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT value FROM marker ORDER BY idempotency_key"
            ).fetchall()
        return tuple(str(row[0]) for row in rows)


class AbruptProcessExit(BaseException):
    """Test-only process loss that bypasses ordinary runtime cleanup."""


class CrashBeforeGovernanceRuntime(OperationRuntime):
    async def _govern_task(
        self,
        operation_id: str,
        task_id: str,
    ) -> GovernanceDecision:
        raise AbruptProcessExit


class DenyMarkerPolicy(DefaultPolicyEvaluator):
    def evaluate(
        self,
        facts: GovernanceFacts,
        *,
        evaluated_at: datetime,
    ) -> GovernanceDecision:
        return GovernanceDecision(
            effect=PolicyEffect.DENY,
            code="test_policy_denied",
            reason="The test policy denies this exact marker task.",
            task_fingerprint=facts.task_fingerprint,
            policy_fingerprint=self.profile.fingerprint,
            evaluated_at=evaluated_at,
        )


def _registry(executor: DurableMarkerExecutor) -> CapabilityRegistry:
    capability = Capability(
        id="test.marker.write",
        owner="phase2-approval-test",
        description="Set one test-owned durable marker.",
        input_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        output_evidence_kind="test.marker.result",
        output_schema_version=1,
        output_schema={
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        executor_id=executor.executor_id,
        access_mode=AccessMode.WRITE,
        risk=RiskLevel.MEDIUM,
        side_effecting=True,
        idempotent=True,
        replay_safe=True,
    )
    return CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="set_test_marker",
                capability_id=capability.id,
                description=capability.description,
            ),
        ),
    )


class MarkerDomain:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self.registry = registry

    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]:
        return self.registry.tool_definitions()

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal:
        _, capability = self.registry.resolve_tool(call.name)
        arguments = self.registry.validate_arguments(capability.id, call.arguments)
        return ActionProposal(
            operation_id=operation.operation.id,
            turn_id=operation.turns[-1].id,
            call_id=call.id,
            capability_id=capability.id,
            arguments=arguments,
            proposed_at=NOW,
        )

    async def project_observation(self, evidence: Evidence) -> Observation:
        return Observation(
            operation_id=evidence.operation_id,
            turn_id=evidence.turn_id,
            task_id=evidence.task_id,
            evidence_id=evidence.id,
            code="test.marker.succeeded",
            message="The test marker changed.",
            payload=evidence.payload,
            success=True,
            created_at=NOW,
        )

    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness:
        denied = any(
            observation.code == "approval_denied"
            for observation in operation.observations
        )
        allowed = bool(operation.evidence) or denied
        return Readiness(
            allowed=allowed,
            code="marker_ready" if allowed else "marker_missing",
            message="The marker outcome is explicit and durable.",
            missing_facts=() if allowed else ("marker_evidence",),
            evaluated_at=NOW,
        )


class MarkerContextBuilder:
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest:
        if not operation.model_calls:
            messages = (
                CanonicalMessage(
                    agent_id=operation.operation.agent_id,
                    operation_id=operation.operation.id,
                    turn_id=turn.id,
                    role=MessageRole.USER,
                    content=(TextBlock("Set the marker after approval."),),
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
                task = next(
                    item for item in operation.tasks if item.id == observation.task_id
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


async def _prepare_write(
    runtime: OperationRuntime,
    registry: CapabilityRegistry,
) -> tuple[str, str]:
    trigger = AgentTrigger(
        id="trigger-marker",
        agent_id="agent-marker",
        kind=TriggerKind.USER,
        source_id="user-marker",
        payload={"request": "set marker"},
        created_at=NOW,
    )
    operation = await runtime.begin(trigger)
    turn = await runtime.begin_turn(operation.operation.id)
    tool = registry.tool_definitions()[0]
    _, capability = registry.resolve_tool(tool.name)
    request = ModelRequest(
        operation_id=operation.operation.id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id=trigger.agent_id,
                operation_id=operation.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Set the marker."),),
            ),
        ),
        tools=(tool,),
    )
    model_call = await runtime.begin_model_call(
        operation.operation.id,
        turn.id,
        "mock:scripted",
        request,
    )
    tool_call = ToolCall(
        id="call-marker",
        name=tool.name,
        arguments={"value": "approved-value"},
    )
    await runtime.record_model_response(
        operation.operation.id,
        model_call.id,
        ModelResponse(
            tool_calls=(tool_call,),
            finish_reason=FinishReason.TOOL_CALLS,
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    result = await runtime.submit(
        ActionProposal(
            operation_id=operation.operation.id,
            turn_id=turn.id,
            call_id=tool_call.id,
            capability_id=capability.id,
            arguments=tool_call.arguments,
            proposed_at=NOW,
        )
    )
    assert result is None
    snapshot = await runtime.inspect(operation.operation.id)
    return snapshot.operation.id, snapshot.tasks[0].id


async def test_side_effect_waits_and_decision_mutates_only_approval(
    tmp_path: Path,
) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)

    operation_id, task_id = await _prepare_write(runtime, registry)
    waiting = await runtime.inspect(operation_id)
    assert waiting.operation.status is OperationStatus.WAITING_FOR_APPROVAL
    assert waiting.tasks[0].status is TaskStatus.WAITING_FOR_APPROVAL
    assert len(waiting.approvals) == 1
    approval = waiting.approvals[0]
    assert approval.status is ApprovalStatus.PENDING
    assert waiting.loop_state.waiting_approval_id == approval.id
    assert executor.requests == []
    assert executor.values() == ()

    decided = await runtime.decide_approval(
        approval.id,
        status=ApprovalStatus.APPROVED,
        decided_by="reviewer-1",
        reason="Approved for the isolated test marker.",
    )
    after_decision = await runtime.inspect(operation_id)
    assert decided.status is ApprovalStatus.APPROVED
    assert after_decision.tasks == waiting.tasks
    assert after_decision.operation == waiting.operation
    assert executor.requests == []
    assert executor.values() == ()

    assert await runtime.resume_approval(operation_id)
    evidence = await runtime.resume_task(operation_id, task_id)
    assert evidence is not None
    assert executor.values() == ("approved-value",)

    repeated = await runtime.resume_task(operation_id, task_id)
    assert repeated == evidence
    assert len(executor.requests) == 1
    assert executor.values() == ("approved-value",)


async def test_decision_channel_accepts_only_approve_or_deny(tmp_path: Path) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    operation_id, _ = await _prepare_write(runtime, registry)
    waiting = await runtime.inspect(operation_id)

    with pytest.raises(ValueError, match="approve or deny"):
        await runtime.decide_approval(
            waiting.approvals[0].id,
            status=ApprovalStatus.CANCELLED,
            decided_by="reviewer-1",
            reason="Cancellation belongs to the operation owner.",
        )

    unchanged = await runtime.inspect(operation_id)
    assert unchanged == waiting
    assert executor.requests == []
    assert executor.values() == ()


async def test_policy_denial_is_durable_and_invokes_no_executor(tmp_path: Path) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    runtime = OperationRuntime(
        capabilities=registry,
        policy=DenyMarkerPolicy(),
        clock=lambda: NOW,
    )

    operation_id, task_id = await _prepare_write(runtime, registry)
    denied = await runtime.inspect(operation_id)

    assert denied.operation.status is OperationStatus.RUNNING
    assert denied.tasks[0].id == task_id
    assert denied.tasks[0].status is TaskStatus.FAILED
    assert denied.tasks[0].error_code == "test_policy_denied"
    assert denied.approvals == ()
    assert denied.observations[-1].code == "test_policy_denied"
    assert not denied.observations[-1].success
    assert denied.loop_state.observation_characters > 0
    assert any(event.type == "governance.denied" for event in denied.events)
    assert executor.requests == []
    assert executor.values() == ()


async def test_denial_is_visible_and_invokes_no_executor(tmp_path: Path) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    operation_id, _ = await _prepare_write(runtime, registry)
    waiting = await runtime.inspect(operation_id)

    await runtime.decide_approval(
        waiting.approvals[0].id,
        status=ApprovalStatus.DENIED,
        decided_by="reviewer-1",
        reason="Denied by the reviewer.",
    )
    assert await runtime.resume_approval(operation_id)
    denied = await runtime.inspect(operation_id)

    assert denied.operation.status is OperationStatus.RUNNING
    assert denied.tasks[0].status is TaskStatus.FAILED
    assert denied.tasks[0].error_code == "approval_denied"
    assert denied.observations[-1].code == "approval_denied"
    assert not denied.observations[-1].success
    assert denied.observations[-1].task_id == denied.tasks[0].id
    assert executor.requests == []
    assert executor.values() == ()


async def test_loop_wakes_and_resumes_the_same_operation_after_approval(
    tmp_path: Path,
) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="call-loop-marker",
                        name="set_test_marker",
                        arguments={"value": "approved-value"},
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            ModelResponse(
                text="The approved marker was changed.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=MarkerContextBuilder(),
        domain=MarkerDomain(registry),
    )
    trigger = AgentTrigger(
        id="trigger-loop-marker",
        agent_id="agent-marker",
        kind=TriggerKind.USER,
        source_id="user-marker",
        payload={"request": "set marker"},
        created_at=NOW,
    )

    waiting_exit = await loop.run(trigger)
    waiting = await runtime.inspect(waiting_exit.operation_id)
    assert waiting_exit.kind is LoopExitKind.WAITING
    assert waiting.operation.status is OperationStatus.WAITING_FOR_APPROVAL
    assert len(provider.requests) == 1
    assert executor.values() == ()

    await runtime.decide_approval(
        waiting.approvals[0].id,
        status=ApprovalStatus.APPROVED,
        decided_by="reviewer-1",
        reason="Approved for the test-owned marker.",
    )
    assert executor.values() == ()
    completed_exit = await loop.resume(waiting.operation.id)
    completed = await runtime.inspect(waiting.operation.id)

    assert completed_exit.kind is LoopExitKind.COMPLETED
    assert completed.operation.id == waiting.operation.id
    assert completed.tasks[0].status is TaskStatus.SUCCEEDED
    assert completed.approvals[0].status is ApprovalStatus.APPROVED
    assert executor.values() == ("approved-value",)
    assert len(executor.requests) == 1
    assert len(provider.requests) == 2
    provider.assert_consumed()


async def test_loop_projects_denial_to_the_model_without_executor_io(
    tmp_path: Path,
) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    runtime = OperationRuntime(capabilities=registry, clock=lambda: NOW)
    provider = MockModelProvider(
        (
            ModelResponse(
                tool_calls=(
                    ToolCall(
                        id="call-loop-denied",
                        name="set_test_marker",
                        arguments={"value": "forbidden-value"},
                    ),
                ),
                finish_reason=FinishReason.TOOL_CALLS,
            ),
            ModelResponse(
                text="The marker change was denied.",
                finish_reason=FinishReason.STOP,
            ),
        )
    )
    loop = AgentLoop(
        runtime=runtime,
        model=provider,
        context_builder=MarkerContextBuilder(),
        domain=MarkerDomain(registry),
    )
    trigger = AgentTrigger(
        id="trigger-loop-denied",
        agent_id="agent-marker",
        kind=TriggerKind.USER,
        source_id="user-marker",
        payload={"request": "set marker"},
        created_at=NOW,
    )

    waiting_exit = await loop.run(trigger)
    waiting = await runtime.inspect(waiting_exit.operation_id)
    await runtime.decide_approval(
        waiting.approvals[0].id,
        status=ApprovalStatus.DENIED,
        decided_by="reviewer-1",
        reason="The reviewer denied this marker change.",
    )
    completed_exit = await loop.resume(waiting.operation.id)
    completed = await runtime.inspect(waiting.operation.id)

    assert completed_exit.kind is LoopExitKind.COMPLETED
    assert completed.tasks[0].status is TaskStatus.FAILED
    assert completed.observations[0].code == "approval_denied"
    assert executor.requests == []
    assert executor.values() == ()
    second_request = provider.requests[1]
    result_block = second_request.messages[-1].content[0]
    assert isinstance(result_block, ToolResultBlock)
    assert result_block.is_error
    assert result_block.output["status"] == "denied"
    provider.assert_consumed()


async def test_resume_rejects_a_changed_policy_fingerprint(tmp_path: Path) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    store = InMemoryOperationStore(clock=lambda: NOW)
    requesting_runtime = OperationRuntime(
        capabilities=registry,
        store=store,
        clock=lambda: NOW,
    )
    operation_id, _ = await _prepare_write(requesting_runtime, registry)
    waiting = await requesting_runtime.inspect(operation_id)
    await requesting_runtime.decide_approval(
        waiting.approvals[0].id,
        status=ApprovalStatus.APPROVED,
        decided_by="reviewer-1",
        reason="Approval is bound to policy version one.",
    )
    changed_policy_runtime = OperationRuntime(
        capabilities=registry,
        store=store,
        policy=DefaultPolicyEvaluator(DefaultPolicyProfile(version="2")),
        clock=lambda: NOW,
    )

    with pytest.raises(OperationStateError, match="exact task and policy"):
        await changed_policy_runtime.resume_approval(operation_id)
    assert executor.requests == []
    assert executor.values() == ()


async def test_concurrent_resume_changes_the_marker_once(tmp_path: Path) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    store = InMemoryOperationStore(clock=lambda: NOW)
    first_runtime = OperationRuntime(
        capabilities=registry,
        store=store,
        clock=lambda: NOW,
        lease_holder_id="holder-first",
    )
    second_runtime = OperationRuntime(
        capabilities=registry,
        store=store,
        clock=lambda: NOW,
        lease_holder_id="holder-second",
    )
    operation_id, task_id = await _prepare_write(first_runtime, registry)
    waiting = await first_runtime.inspect(operation_id)
    await first_runtime.decide_approval(
        waiting.approvals[0].id,
        status=ApprovalStatus.APPROVED,
        decided_by="reviewer-1",
        reason="Approved once.",
    )

    reconciled = await asyncio.gather(
        first_runtime.resume_approval(operation_id),
        second_runtime.resume_approval(operation_id),
    )
    assert sorted(reconciled) == [False, True]
    results = await asyncio.gather(
        first_runtime.resume_task(operation_id, task_id),
        second_runtime.resume_task(operation_id, task_id),
        return_exceptions=True,
    )
    assert any(isinstance(result, Evidence) for result in results)
    assert len(executor.requests) == 1
    assert executor.values() == ("approved-value",)


async def test_approval_decision_racing_cancellation_has_one_terminal_transition(
    tmp_path: Path,
) -> None:
    executor = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(executor)
    store = InMemoryOperationStore(clock=lambda: NOW)
    decision_runtime = OperationRuntime(
        capabilities=registry,
        store=store,
        clock=lambda: NOW,
    )
    cancellation_runtime = OperationRuntime(
        capabilities=registry,
        store=store,
        clock=lambda: NOW,
    )
    operation_id, _ = await _prepare_write(decision_runtime, registry)
    waiting = await decision_runtime.inspect(operation_id)
    approval_id = waiting.approvals[0].id

    await asyncio.gather(
        decision_runtime.decide_approval(
            approval_id,
            status=ApprovalStatus.APPROVED,
            decided_by="reviewer-1",
            reason="Concurrent approval.",
        ),
        cancellation_runtime.interrupt(operation_id, "concurrent_cancel"),
        return_exceptions=True,
    )
    final = await decision_runtime.inspect(operation_id)
    terminal_events = [
        event
        for event in final.events
        if event.approval_id == approval_id
        and event.type in {"approval.approved", "approval.cancelled"}
    ]

    assert final.operation.status is OperationStatus.INTERRUPTED
    assert final.approvals[0].status in {
        ApprovalStatus.APPROVED,
        ApprovalStatus.CANCELLED,
    }
    assert len(terminal_events) == 1
    assert terminal_events[0].type == (f"approval.{final.approvals[0].status.value}")
    assert executor.requests == []
    assert executor.values() == ()


async def test_approved_wait_survives_sqlite_reopen_and_reuses_exact_task(
    tmp_path: Path,
) -> None:
    marker = DurableMarkerExecutor(tmp_path / "marker.sqlite3")
    registry = _registry(marker)
    state_path = tmp_path / "state.sqlite3"
    first_store = await SQLiteOperationStore.open(state_path, clock=lambda: NOW)
    first_runtime = OperationRuntime(
        capabilities=registry,
        store=first_store,
        clock=lambda: NOW,
    )
    operation_id, task_id = await _prepare_write(first_runtime, registry)
    waiting = await first_runtime.inspect(operation_id)
    await first_runtime.decide_approval(
        waiting.approvals[0].id,
        status=ApprovalStatus.APPROVED,
        decided_by="reviewer-1",
        reason="Approved before restart.",
    )
    await first_store.close()

    reopened_store = await SQLiteOperationStore.open(state_path, clock=lambda: NOW)
    reopened_runtime = OperationRuntime(
        capabilities=registry,
        store=reopened_store,
        clock=lambda: NOW,
    )
    try:
        assert await reopened_runtime.resume_approval(operation_id)
        evidence = await reopened_runtime.resume_task(operation_id, task_id)
        assert evidence is not None
        assert evidence.operation_id == operation_id
        assert evidence.task_id == task_id
        assert marker.values() == ("approved-value",)
        assert await reopened_runtime.resume_task(operation_id, task_id) == evidence
        assert len(marker.requests) == 1
    finally:
        await reopened_store.close()


async def test_pending_side_effect_is_regoverned_after_restart_before_executor_io(
    tmp_path: Path,
) -> None:
    marker_path = tmp_path / "marker.sqlite3"
    state_path = tmp_path / "state.sqlite3"
    first_executor = DurableMarkerExecutor(marker_path)
    first_registry = _registry(first_executor)
    first_store = await SQLiteOperationStore.open(state_path, clock=lambda: NOW)
    try:
        first_runtime = CrashBeforeGovernanceRuntime(
            capabilities=first_registry,
            store=first_store,
            clock=lambda: NOW,
        )

        with pytest.raises(AbruptProcessExit):
            await _prepare_write(first_runtime, first_registry)

        before = await first_store.load_by_trigger("trigger-marker")
        assert before is not None
        assert before.snapshot.tasks[0].status is TaskStatus.PENDING
        assert before.snapshot.approvals == ()
        assert first_executor.requests == []
        assert first_executor.values() == ()
    finally:
        await first_store.close()

    resumed_executor = DurableMarkerExecutor(marker_path)
    resumed_registry = _registry(resumed_executor)
    resumed_store = await SQLiteOperationStore.open(state_path, clock=lambda: NOW)
    try:
        resumed_runtime = OperationRuntime(
            capabilities=resumed_registry,
            store=resumed_store,
            clock=lambda: NOW,
        )
        resumed_loop = AgentLoop(
            runtime=resumed_runtime,
            model=MockModelProvider(()),
            context_builder=MarkerContextBuilder(),
            domain=MarkerDomain(resumed_registry),
        )

        exits = await resumed_loop.recover_startup("agent-marker")
        waiting = await resumed_runtime.inspect(before.snapshot.operation.id)
    finally:
        await resumed_store.close()

    assert tuple(exit.kind for exit in exits) == (LoopExitKind.WAITING,)
    assert exits[0].reason == "waiting_for_approval"
    assert waiting.operation.status is OperationStatus.WAITING_FOR_APPROVAL
    assert waiting.tasks[0].id == before.snapshot.tasks[0].id
    assert waiting.tasks[0].status is TaskStatus.WAITING_FOR_APPROVAL
    assert len(waiting.approvals) == 1
    assert waiting.approvals[0].status is ApprovalStatus.PENDING
    assert waiting.approvals[0].task_id == waiting.tasks[0].id
    assert first_executor.requests == []
    assert resumed_executor.requests == []
    assert resumed_executor.values() == ()
