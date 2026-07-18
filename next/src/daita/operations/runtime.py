"""In-memory operation runtime used by the Phase 1 loop laboratory."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import math
from typing import TypeVar
from uuid import uuid4

from .._json import canonical_json
from ..capabilities import (
    Capability,
    CapabilityExecutionError,
    CapabilityRegistry,
    EvidenceCandidate,
    EvidenceValidationError,
    ExecutionRequest,
    Executor,
)
from ..events.models import RuntimeEvent
from ..llm.models import ModelRequest, ModelResponse, ToolCall
from ..loop.models import (
    LoopBudgets,
    LoopExit,
    LoopExitKind,
    LoopPhase,
    LoopState,
    Readiness,
    Turn,
)
from ..storage.blobs import BlobMetadata, BlobPut, BlobStore
from .checkpoints import ModelCall, ModelCallStatus, OperationSnapshot
from .governance import (
    ApprovalRequest,
    ApprovalStatus,
    DefaultPolicyEvaluator,
    GovernanceDecision,
    GovernanceFacts,
    PolicyEffect,
)
from .models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Evidence,
    Observation,
    Operation,
    OperationStatus,
    Task,
    TaskDependency,
    TaskExecutionFacts,
    TaskStatus,
    TriggerKind,
)
from .leases import TaskClaimRequest, TaskLease, TaskLeaseGuard
from .store import (
    ExpiredTaskLeaseError,
    InMemoryOperationStore,
    InvalidOperationCheckpointError,
    OperationAlreadyExistsError,
    OperationNotFoundError,
    OperationRevisionConflict,
    OperationStoreError,
    StaleTaskFenceError,
    TriggerAlreadyClaimedError,
    TaskExecutionStore,
)

_T = TypeVar("_T")


class OperationStateError(RuntimeError):
    """Raised when a runtime transition conflicts with committed state."""


class TaskExecutionTimeout(CapabilityExecutionError):
    """Raised after a runtime-owned executor deadline is durably recorded."""

    def __init__(self, task_id: str, timeout_seconds: float) -> None:
        self.task_id = task_id
        self.timeout_seconds = timeout_seconds
        super().__init__(f"task {task_id} exceeded {timeout_seconds:g} seconds")


class OperationWallTimeExceeded(CapabilityExecutionError):
    """Raised when the operation deadline prevents or bounds executor I/O."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        super().__init__(
            f"operation wall-time limit expired before task {task_id} completed"
        )


@dataclass(slots=True)
class _OperationState:
    revision: int
    trigger: AgentTrigger
    operation: Operation
    loop_state: LoopState
    budgets: LoopBudgets
    turns: list[Turn]
    model_calls: list[ModelCall]
    readiness: list[Readiness]
    tasks: list[Task]
    task_dependencies: list[TaskDependency]
    task_leases: list[TaskLease]
    approvals: list[ApprovalRequest]
    evidence: list[Evidence]
    observations: list[Observation]
    events: list[RuntimeEvent]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _random_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


async def _await_store_write(awaitable: Awaitable[_T]) -> tuple[_T, bool]:
    """Resolve one atomic store write before propagating caller cancellation."""

    write = asyncio.ensure_future(awaitable)
    cancellation_requested = False
    while not write.done():
        try:
            await asyncio.shield(write)
        except asyncio.CancelledError:
            cancellation_requested = True
            continue
        except BaseException:
            # Resolve the completed write below so a caller cancellation that
            # arrived first retains precedence without losing the write error.
            break
    try:
        result = write.result()
    except BaseException as error:
        if cancellation_requested and not isinstance(error, asyncio.CancelledError):
            raise asyncio.CancelledError from error
        raise
    return result, cancellation_requested


async def _await_resistant_task(task: asyncio.Task[_T]) -> _T:
    """Wait for required cleanup even when cancellation is requested again."""

    while not task.done():
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            continue
    return task.result()


class OperationRuntime:
    """Commit inspectable operation/loop transitions through one repository."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _random_id,
        capabilities: CapabilityRegistry | None = None,
        store: TaskExecutionStore | None = None,
        blob_store: BlobStore | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        lease_holder_id: str | None = None,
        lease_duration_seconds: float = 60.0,
    ) -> None:
        if lease_holder_id is not None:
            _required_text(lease_holder_id, "lease_holder_id")
        if (
            not isinstance(lease_duration_seconds, (int, float))
            or isinstance(lease_duration_seconds, bool)
            or not math.isfinite(lease_duration_seconds)
            or lease_duration_seconds <= 0
        ):
            raise ValueError("lease_duration_seconds must be finite and positive")
        self._clock = clock
        self._id_factory = id_factory
        self._capabilities = capabilities or CapabilityRegistry()
        self._lock = asyncio.Lock()
        self._store = (
            store if store is not None else InMemoryOperationStore(clock=clock)
        )
        self._blob_store = blob_store
        self._policy = policy or DefaultPolicyEvaluator()
        self._lease_holder_id = lease_holder_id or _random_id("runtime-holder")
        self._lease_duration_seconds = float(lease_duration_seconds)

    async def begin(
        self,
        trigger: AgentTrigger,
        *,
        budgets: LoopBudgets = LoopBudgets(),
    ) -> OperationSnapshot:
        cancellation_requested = False
        async with self._lock:
            if not isinstance(budgets, LoopBudgets):
                raise TypeError("budgets must be a LoopBudgets record")
            if trigger.kind is TriggerKind.EVENT:
                raise ValueError("event triggers are reserved for a later phase")

            now = self._clock()
            operation_id = self._id_factory("operation")
            operation = Operation(
                id=operation_id,
                agent_id=trigger.agent_id,
                trigger_id=trigger.id,
                session_id=trigger.session_id,
                status=OperationStatus.RUNNING,
                created_at=now,
                updated_at=now,
            )
            state = _OperationState(
                revision=0,
                trigger=trigger,
                operation=operation,
                loop_state=LoopState(phase=LoopPhase.PREPARING_CONTEXT),
                budgets=budgets,
                turns=[],
                model_calls=[],
                readiness=[],
                tasks=[],
                task_dependencies=[],
                task_leases=[],
                approvals=[],
                evidence=[],
                observations=[],
                events=[],
            )
            self._append_event(state, "trigger.received")
            self._append_event(state, "operation.created")
            try:
                committed, cancellation_requested = await _await_store_write(
                    self._store.create(self._snapshot(state))
                )
            except TriggerAlreadyClaimedError as error:
                existing = await self._store.load_by_trigger(trigger.id)
                if existing is not None and existing.snapshot.trigger == trigger:
                    return existing.snapshot
                existing_id = (
                    "unknown" if existing is None else existing.snapshot.operation.id
                )
                raise OperationStateError(
                    f"trigger already owns a different operation input: {existing_id}"
                ) from error
            except OperationAlreadyExistsError as error:
                raise OperationStateError(
                    f"operation already exists: {operation_id}"
                ) from error
            snapshot = committed.operation.snapshot

        if cancellation_requested:

            async def interrupt_or_accept_terminal_race() -> None:
                try:
                    await self.interrupt(operation_id)
                except OperationStateError:
                    snapshot = await self.inspect(operation_id)
                    if snapshot.operation.status not in {
                        OperationStatus.SUCCEEDED,
                        OperationStatus.FAILED,
                        OperationStatus.CANCELLED,
                        OperationStatus.INTERRUPTED,
                    }:
                        raise

            interruption = asyncio.create_task(interrupt_or_accept_terminal_race())
            await _await_resistant_task(interruption)
            raise asyncio.CancelledError
        return snapshot

    async def begin_turn(self, operation_id: str) -> Turn:
        async with self._lock:
            state = await self._working_state(operation_id)
            now = self._clock()
            turn = Turn(
                id=self._id_factory("turn"),
                operation_id=operation_id,
                number=state.loop_state.turn_count + 1,
                created_at=now,
            )
            state.turns.append(turn)
            state.loop_state = replace(
                state.loop_state,
                phase=LoopPhase.PREPARING_CONTEXT,
                turn_count=turn.number,
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(state, "turn.created", turn_id=turn.id)
            await self._commit(state)
            return turn

    async def begin_model_call(
        self,
        operation_id: str,
        turn_id: str,
        provider_id: str,
        request: ModelRequest,
    ) -> ModelCall:
        async with self._lock:
            _required_text(provider_id, "provider_id")
            if not isinstance(request, ModelRequest):
                raise TypeError("request must be a ModelRequest")
            state = await self._working_state(operation_id)
            turn_index, turn = self._turn(state, turn_id)
            if request.operation_id != operation_id or request.turn_id != turn_id:
                raise OperationStateError(
                    "model request linkage does not match the turn"
                )
            if turn.model_request_id is not None:
                raise OperationStateError("turn already has a model request")
            if any(
                message.agent_id != state.operation.agent_id
                for message in request.messages
            ):
                raise OperationStateError(
                    "model request messages do not belong to the operation agent"
                )

            now = self._clock()
            model_call = ModelCall(
                id=self._id_factory("model-call"),
                operation_id=operation_id,
                turn_id=turn_id,
                provider_id=provider_id,
                request=request,
                status=ModelCallStatus.STARTED,
                created_at=now,
                updated_at=now,
            )
            state.model_calls.append(model_call)
            state.turns[turn_index] = replace(
                turn,
                model_request_id=model_call.id,
            )
            state.loop_state = replace(
                state.loop_state,
                phase=LoopPhase.AWAITING_MODEL,
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "context.built",
                turn_id=turn_id,
                model_call_id=model_call.id,
            )
            self._append_event(
                state,
                "model_call.started",
                turn_id=turn_id,
                model_call_id=model_call.id,
                payload={"model_call_id": model_call.id, "provider_id": provider_id},
            )
            await self._commit(state)
            return model_call

    async def record_model_response(
        self,
        operation_id: str,
        model_call_id: str,
        response: ModelResponse,
        *,
        next_phase: LoopPhase,
    ) -> None:
        async with self._lock:
            if not isinstance(response, ModelResponse):
                raise TypeError("response must be a ModelResponse")
            state = await self._working_state(operation_id)
            call_index, model_call = self._model_call(state, model_call_id)
            if model_call.status is not ModelCallStatus.STARTED:
                raise OperationStateError("model call is already terminal")
            if next_phase not in {
                LoopPhase.VALIDATING_ACTION,
                LoopPhase.SYNTHESIZING,
            }:
                raise OperationStateError("invalid post-model loop phase")

            now = self._clock()
            state.model_calls[call_index] = replace(
                model_call,
                status=ModelCallStatus.COMPLETED,
                response=response,
                updated_at=now,
            )
            turn_index, turn = self._turn(state, model_call.turn_id)
            state.turns[turn_index] = replace(turn, model_response_id=model_call.id)
            state.loop_state = replace(
                state.loop_state,
                phase=next_phase,
                input_tokens=state.loop_state.input_tokens
                + response.usage.input_tokens,
                output_tokens=state.loop_state.output_tokens
                + response.usage.output_tokens,
                estimated_cost_usd=state.loop_state.estimated_cost_usd
                + response.usage.estimated_cost_usd,
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "model_response.recorded",
                turn_id=model_call.turn_id,
                model_call_id=model_call.id,
                payload={
                    "finish_reason": response.finish_reason.value,
                    "model_call_id": model_call.id,
                    "total_tokens": response.usage.total_tokens,
                    "estimated_cost_usd": str(response.usage.estimated_cost_usd),
                },
            )
            await self._commit(state)

    async def record_readiness(
        self,
        operation_id: str,
        final_text: str,
        readiness: Readiness,
    ) -> Observation | None:
        async with self._lock:
            _required_text(final_text, "final_text")
            if not isinstance(readiness, Readiness):
                raise TypeError("readiness must be a Readiness record")
            state = await self._working_state(operation_id)
            if not state.model_calls or (
                state.model_calls[-1].status is not ModelCallStatus.COMPLETED
            ):
                raise OperationStateError("readiness requires a completed model call")
            model_call = state.model_calls[-1]
            model_response = model_call.response
            if (
                model_response is None
                or model_response.tool_calls
                or model_response.text != final_text
            ):
                raise OperationStateError(
                    "readiness text must match the committed final model response"
                )
            readiness_turn_id = model_call.turn_id
            if any(
                event.type == "readiness.recorded"
                and event.turn_id == readiness_turn_id
                for event in state.events
            ):
                raise OperationStateError(
                    "model response already has a readiness decision"
                )
            if any(
                task.status
                in {
                    TaskStatus.PENDING,
                    TaskStatus.READY,
                    TaskStatus.CLAIMED,
                    TaskStatus.RUNNING,
                    TaskStatus.WAITING_FOR_APPROVAL,
                }
                for task in state.tasks
            ):
                raise OperationStateError(
                    "readiness requires every task to be terminal"
                )
            if any(
                not any(
                    observation.evidence_id == evidence.id
                    for observation in state.observations
                )
                for evidence in state.evidence
                if evidence.accepted
            ):
                raise OperationStateError(
                    "readiness requires every accepted evidence item to be observed"
                )
            now = self._clock()
            state.readiness.append(readiness)
            correction: Observation | None = None
            if not readiness.allowed:
                correction = Observation(
                    operation_id=operation_id,
                    turn_id=model_call.turn_id,
                    code=readiness.code,
                    message=readiness.message,
                    payload={"missing_facts": readiness.missing_facts},
                    success=False,
                    created_at=now,
                )
                state.observations.append(correction)
            state.loop_state = replace(
                state.loop_state,
                phase=(
                    LoopPhase.SYNTHESIZING if readiness.allowed else LoopPhase.OBSERVING
                ),
                repair_count=(
                    state.loop_state.repair_count + (0 if correction is None else 1)
                ),
                observation_characters=(
                    state.loop_state.observation_characters
                    + (
                        0
                        if correction is None
                        else self._observation_characters(correction)
                    )
                ),
                final_answer_candidate=(final_text if readiness.allowed else None),
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "readiness.recorded",
                turn_id=model_call.turn_id,
                model_call_id=model_call.id,
                payload={"allowed": readiness.allowed, "code": readiness.code},
            )
            if correction is not None:
                self._append_event(
                    state,
                    "observation.recorded",
                    turn_id=correction.turn_id,
                    model_call_id=model_call.id,
                    payload={
                        "code": correction.code,
                        "repair": "readiness",
                        "truncated": correction.truncated,
                    },
                )
            await self._commit(state)
            return correction

    async def record_action_rejection(
        self,
        operation_id: str,
        turn_id: str,
        call: ToolCall,
        rejection: ActionRejection,
    ) -> Observation:
        if not isinstance(call, ToolCall):
            raise TypeError("call must be a ToolCall record")
        if not isinstance(rejection, ActionRejection):
            raise TypeError("rejection must be an ActionRejection record")

        async with self._lock:
            state = await self._working_state(operation_id)
            model_call, committed_call = self._committed_tool_call(
                state,
                turn_id,
                call.id,
            )
            if committed_call != call:
                raise OperationStateError(
                    "rejected action does not match the committed tool call"
                )
            fingerprint = self._action_fingerprint(committed_call)
            self._require_preceding_calls_observed(
                state,
                model_call,
                committed_call,
            )
            assert model_call.response is not None
            call_position = model_call.response.tool_calls.index(committed_call)
            affected_calls = model_call.response.tool_calls[call_position:]
            for affected_call in affected_calls:
                if any(
                    observation.turn_id == turn_id
                    and observation.call_id == affected_call.id
                    for observation in state.observations
                ):
                    raise OperationStateError("tool call already has an observation")
                if any(
                    task.turn_id == turn_id and task.call_id == affected_call.id
                    for task in state.tasks
                ):
                    raise OperationStateError(
                        "materialized task cannot be rejected or skipped"
                    )

            now = self._clock()
            observation = Observation(
                operation_id=operation_id,
                turn_id=turn_id,
                call_id=call.id,
                code=rejection.code,
                message=rejection.message,
                payload=rejection.details,
                success=False,
                created_at=now,
            )
            skipped_observations = tuple(
                Observation(
                    operation_id=operation_id,
                    turn_id=turn_id,
                    call_id=skipped_call.id,
                    code="action.skipped_after_rejection",
                    message=(
                        "Tool call was not run because an earlier call was rejected."
                    ),
                    payload={
                        "blocked_by_call_id": call.id,
                        "blocked_by_code": rejection.code,
                    },
                    success=False,
                    created_at=now,
                )
                for skipped_call in affected_calls[1:]
            )
            last_fingerprint = (
                state.loop_state.no_progress_fingerprints[-1]
                if state.loop_state.no_progress_fingerprints
                else None
            )
            identical_count = (
                state.loop_state.identical_failure_count + 1
                if last_fingerprint == fingerprint
                else 1
            )
            state.observations.append(observation)
            state.observations.extend(skipped_observations)
            state.loop_state = replace(
                state.loop_state,
                phase=LoopPhase.OBSERVING,
                repair_count=state.loop_state.repair_count + 1,
                identical_failure_count=identical_count,
                observation_characters=(
                    state.loop_state.observation_characters
                    + self._observation_characters(observation)
                    + sum(
                        self._observation_characters(skipped)
                        for skipped in skipped_observations
                    )
                ),
                no_progress_fingerprints=(
                    state.loop_state.no_progress_fingerprints
                    if last_fingerprint == fingerprint
                    else (*state.loop_state.no_progress_fingerprints, fingerprint)
                ),
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "action.rejected",
                turn_id=turn_id,
                model_call_id=model_call.id,
                call_id=call.id,
                payload={
                    "code": rejection.code,
                    "fingerprint": fingerprint,
                    "tool_name": call.name,
                    "model_call_id": model_call.id,
                },
            )
            self._append_event(
                state,
                "observation.recorded",
                turn_id=turn_id,
                model_call_id=model_call.id,
                call_id=call.id,
                payload={
                    "code": observation.code,
                    "fingerprint": fingerprint,
                    "repair": "action",
                    "truncated": observation.truncated,
                },
            )
            for skipped_call, skipped in zip(
                affected_calls[1:],
                skipped_observations,
                strict=True,
            ):
                self._append_event(
                    state,
                    "action.skipped",
                    turn_id=turn_id,
                    model_call_id=model_call.id,
                    call_id=skipped_call.id,
                    payload={
                        "blocked_by_call_id": call.id,
                        "blocked_by_code": rejection.code,
                        "model_call_id": model_call.id,
                        "tool_name": skipped_call.name,
                    },
                )
                self._append_event(
                    state,
                    "observation.recorded",
                    turn_id=turn_id,
                    model_call_id=model_call.id,
                    call_id=skipped_call.id,
                    payload={
                        "code": skipped.code,
                        "repair": "action_skip",
                        "truncated": skipped.truncated,
                    },
                )
            await self._commit(state)
            return observation

    async def submit(
        self,
        proposal: ActionProposal,
        *,
        timeout_seconds: float | None = None,
    ) -> Evidence | None:
        if not isinstance(proposal, ActionProposal):
            raise TypeError("proposal must be an ActionProposal")
        if timeout_seconds is not None and (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        requested_timeout_seconds = (
            None if timeout_seconds is None else float(timeout_seconds)
        )

        task = await self._materialize_task(proposal)
        decision = await self._govern_task(task.operation_id, task.id)
        if decision.effect is not PolicyEffect.ALLOW:
            return None
        return await self._execute_materialized_task(
            task.operation_id,
            task.id,
            requested_timeout_seconds=requested_timeout_seconds,
        )

    async def _govern_task(
        self,
        operation_id: str,
        task_id: str,
    ) -> GovernanceDecision:
        """Persist the policy outcome before any task can reach executor I/O."""

        async with self._lock:
            state = await self._working_state(operation_id)
            task_index, task = self._task(state, task_id)
            if task.status is not TaskStatus.PENDING:
                raise OperationStateError("only a pending task can enter governance")
            self._resolve_task_execution(task)
            facts = self._governance_facts(state, task)
            now = self._clock()
            decision = self._policy.evaluate(facts, evaluated_at=now)
            model_call = self._completed_model_call_for_turn(state, task.turn_id)
            event_fields = {
                "turn_id": task.turn_id,
                "model_call_id": model_call.id,
                "call_id": task.call_id,
                "task_id": task.id,
                "capability_id": task.capability_id,
                "executor_id": task.executor_id,
            }
            payload = {
                "code": decision.code,
                "policy_fingerprint": decision.policy_fingerprint,
                "task_fingerprint": decision.task_fingerprint,
            }

            if decision.effect is PolicyEffect.ALLOW:
                ready_task = replace(
                    task,
                    status=TaskStatus.READY,
                    updated_at=now,
                )
                state.tasks[task_index] = ready_task
                self._append_event(
                    state,
                    "governance.allowed",
                    **event_fields,
                    payload=payload,
                )
                self._append_event(
                    state,
                    "task.ready",
                    **event_fields,
                    payload={"task_id": task.id},
                )
            elif decision.effect is PolicyEffect.REQUIRE_APPROVAL:
                approval = ApprovalRequest(
                    id=self._id_factory("approval"),
                    operation_id=operation_id,
                    task_id=task.id,
                    task_fingerprint=decision.task_fingerprint,
                    policy_fingerprint=decision.policy_fingerprint,
                    requested_at=now,
                )
                state.approvals.append(approval)
                state.tasks[task_index] = replace(
                    task,
                    status=TaskStatus.WAITING_FOR_APPROVAL,
                    updated_at=now,
                )
                state.operation = replace(
                    state.operation,
                    status=OperationStatus.WAITING_FOR_APPROVAL,
                    updated_at=now,
                )
                state.loop_state = replace(
                    state.loop_state,
                    phase=LoopPhase.AWAITING_APPROVAL,
                    waiting_approval_id=approval.id,
                )
                self._append_event(
                    state,
                    "governance.approval_required",
                    **event_fields,
                    approval_id=approval.id,
                    payload={**payload, "approval_id": approval.id},
                )
                self._append_event(
                    state,
                    "approval.requested",
                    **event_fields,
                    approval_id=approval.id,
                    payload={
                        "approval_id": approval.id,
                        "policy_fingerprint": decision.policy_fingerprint,
                        "task_fingerprint": decision.task_fingerprint,
                    },
                )
            else:
                failed = replace(
                    task,
                    status=TaskStatus.FAILED,
                    updated_at=now,
                    error_code=decision.code,
                )
                state.tasks[task_index] = failed
                observation = Observation(
                    operation_id=operation_id,
                    turn_id=task.turn_id,
                    call_id=task.call_id,
                    task_id=task.id,
                    code=decision.code,
                    message=decision.reason,
                    payload=payload,
                    success=False,
                    created_at=now,
                )
                state.observations.append(observation)
                state.loop_state = replace(
                    state.loop_state,
                    phase=LoopPhase.OBSERVING,
                    observation_characters=(
                        state.loop_state.observation_characters
                        + self._observation_characters(observation)
                    ),
                )
                self._append_event(
                    state,
                    "governance.denied",
                    **event_fields,
                    payload=payload,
                )
                self._append_event(
                    state,
                    "task.failed",
                    **event_fields,
                    payload={"task_id": task.id, "error_code": decision.code},
                )
                self._append_event(
                    state,
                    "observation.recorded",
                    **event_fields,
                    payload={"code": decision.code, "success": False},
                )

            state.operation = replace(state.operation, updated_at=now)
            await self._commit(state)
            return decision

    async def decide_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        decided_by: str,
        reason: str,
    ) -> ApprovalRequest:
        """CAS one pending approval; never execute or resume its task."""

        _required_text(approval_id, "approval_id")
        _required_text(decided_by, "decided_by")
        _required_text(reason, "approval reason")
        if not isinstance(status, ApprovalStatus):
            raise TypeError("status must be an ApprovalStatus")
        if status not in {ApprovalStatus.APPROVED, ApprovalStatus.DENIED}:
            raise ValueError("approval decision must approve or deny")

        async with self._lock:
            committed = await self._store.load_by_approval(approval_id)
            if committed is None:
                raise KeyError(f"Unknown approval: {approval_id}")
            state = self._state_from_snapshot(
                committed.snapshot,
                revision=committed.revision,
            )
            approval_index, approval = self._approval(state, approval_id)
            if approval.status is not ApprovalStatus.PENDING:
                if (
                    approval.status is status
                    and approval.decided_by == decided_by
                    and approval.decision_reason == reason
                ):
                    return approval
                raise OperationStateError(
                    f"approval is already {approval.status.value}: {approval_id}"
                )
            if state.operation.status in {
                OperationStatus.SUCCEEDED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
                OperationStatus.INTERRUPTED,
            }:
                raise OperationStateError("approval operation is already terminal")

            now = self._clock()
            decided = replace(
                approval,
                status=status,
                decided_at=now,
                decided_by=decided_by,
                decision_reason=reason,
            )
            state.approvals[approval_index] = decided
            _, task = self._task(state, approval.task_id)
            model_call = self._completed_model_call_for_turn(state, task.turn_id)
            self._append_event(
                state,
                f"approval.{status.value}",
                turn_id=task.turn_id,
                model_call_id=model_call.id,
                call_id=task.call_id,
                task_id=task.id,
                approval_id=approval.id,
                capability_id=task.capability_id,
                executor_id=task.executor_id,
                payload={
                    "approval_id": approval.id,
                    "decided_by": decided_by,
                    "reason": reason,
                },
            )
            await self._commit(state)
            return decided

    async def resume_approval(self, operation_id: str) -> bool:
        """Apply one decided approval to its exact task without executor I/O."""

        _required_text(operation_id, "operation_id")
        async with self._lock:
            while True:
                state = await self._working_state(operation_id)
                if state.operation.status is not OperationStatus.WAITING_FOR_APPROVAL:
                    return False
                approval_id = state.loop_state.waiting_approval_id
                if approval_id is None:
                    raise OperationStateError(
                        "approval-waiting operation has no approval identity"
                    )
                _, approval = self._approval(state, approval_id)
                if approval.status is ApprovalStatus.PENDING:
                    return False
                task_index, task = self._task(state, approval.task_id)
                if task.status is not TaskStatus.WAITING_FOR_APPROVAL:
                    raise OperationStateError("approval task is not waiting")
                self._resolve_task_execution(task)
                facts = self._governance_facts(state, task)
                current_decision = self._policy.evaluate(
                    facts,
                    evaluated_at=self._clock(),
                )
                if (
                    approval.task_fingerprint != facts.task_fingerprint
                    or approval.policy_fingerprint
                    != current_decision.policy_fingerprint
                    or current_decision.effect is not PolicyEffect.REQUIRE_APPROVAL
                ):
                    raise OperationStateError(
                        "approval no longer matches the exact task and policy"
                    )

                now = self._clock()
                model_call = self._completed_model_call_for_turn(state, task.turn_id)
                event_fields = {
                    "turn_id": task.turn_id,
                    "model_call_id": model_call.id,
                    "call_id": task.call_id,
                    "task_id": task.id,
                    "approval_id": approval.id,
                    "capability_id": task.capability_id,
                    "executor_id": task.executor_id,
                }
                if approval.status is ApprovalStatus.APPROVED:
                    state.tasks[task_index] = replace(
                        task,
                        status=TaskStatus.READY,
                        updated_at=now,
                    )
                    state.loop_state = replace(
                        state.loop_state,
                        phase=LoopPhase.AWAITING_EXECUTION,
                        waiting_approval_id=None,
                    )
                    self._append_event(
                        state,
                        "approval.applied",
                        **event_fields,
                        payload={
                            "approval_id": approval.id,
                            "status": approval.status.value,
                        },
                    )
                    self._append_event(
                        state,
                        "task.ready",
                        **event_fields,
                        payload={"task_id": task.id},
                    )
                else:
                    error_code = f"approval_{approval.status.value}"
                    state.tasks[task_index] = replace(
                        task,
                        status=(
                            TaskStatus.FAILED
                            if approval.status is ApprovalStatus.DENIED
                            else TaskStatus.CANCELLED
                        ),
                        updated_at=now,
                        error_code=(
                            error_code
                            if approval.status is ApprovalStatus.DENIED
                            else None
                        ),
                    )
                    observation = Observation(
                        operation_id=operation_id,
                        turn_id=task.turn_id,
                        call_id=task.call_id,
                        task_id=task.id,
                        code=error_code,
                        message=approval.decision_reason
                        or "The approval was not granted.",
                        payload={
                            "approval_id": approval.id,
                            "status": approval.status.value,
                        },
                        success=False,
                        created_at=now,
                    )
                    state.observations.append(observation)
                    state.loop_state = replace(
                        state.loop_state,
                        phase=LoopPhase.OBSERVING,
                        waiting_approval_id=None,
                        observation_characters=(
                            state.loop_state.observation_characters
                            + self._observation_characters(observation)
                        ),
                    )
                    self._append_event(
                        state,
                        "approval.applied",
                        **event_fields,
                        payload={
                            "approval_id": approval.id,
                            "status": approval.status.value,
                        },
                    )
                    self._append_event(
                        state,
                        (
                            "task.failed"
                            if approval.status is ApprovalStatus.DENIED
                            else "task.cancelled"
                        ),
                        **event_fields,
                        payload={"task_id": task.id, "error_code": error_code},
                    )
                    self._append_event(
                        state,
                        "observation.recorded",
                        **event_fields,
                        payload={"code": error_code, "success": False},
                    )
                state.operation = replace(
                    state.operation,
                    status=OperationStatus.RUNNING,
                    updated_at=now,
                )
                try:
                    await self._commit(state)
                except OperationStateError as error:
                    if isinstance(error.__cause__, OperationRevisionConflict):
                        continue
                    raise
                return True

    async def resume_task(
        self,
        operation_id: str,
        task_id: str,
        *,
        timeout_seconds: float | None = None,
    ) -> Evidence | None:
        """Recover one persisted task and execute only a replay-safe attempt."""

        _required_text(operation_id, "operation_id")
        _required_text(task_id, "task_id")
        if timeout_seconds is not None and (
            not isinstance(timeout_seconds, (int, float))
            or isinstance(timeout_seconds, bool)
            or not math.isfinite(timeout_seconds)
            or timeout_seconds <= 0
        ):
            raise ValueError("timeout_seconds must be finite and positive")
        requested_timeout_seconds = (
            None if timeout_seconds is None else float(timeout_seconds)
        )
        needs_governance = False

        async with self._lock:
            try:
                committed = await self._store.load(operation_id)
            except OperationNotFoundError as error:
                raise KeyError(f"Unknown operation: {operation_id}") from error
            state = self._state_from_snapshot(
                committed.snapshot,
                revision=committed.revision,
            )
            _, task = self._task(state, task_id)

            if task.status is TaskStatus.SUCCEEDED:
                return self._accepted_evidence(state, task.evidence_ids[-1])
            if task.status in {
                TaskStatus.FAILED,
                TaskStatus.CANCELLED,
                TaskStatus.MANUAL_RECOVERY_REQUIRED,
                TaskStatus.WAITING_FOR_APPROVAL,
            }:
                return None
            if state.operation.status in {
                OperationStatus.SUCCEEDED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
                OperationStatus.INTERRUPTED,
            }:
                raise OperationStateError("operation is already terminal")

            if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                active_lease = next(
                    (
                        lease
                        for lease in reversed(state.task_leases)
                        if lease.task_id == task.id and lease.released_at is None
                    ),
                    None,
                )
                if active_lease is None:
                    raise OperationStateError(
                        f"task has no active recovery lease: {task.id}"
                    )
                guard = TaskLeaseGuard(
                    operation_id=operation_id,
                    task_id=task.id,
                    holder_id=active_lease.holder_id,
                    attempt=active_lease.attempt,
                    fencing_token=active_lease.fencing_token,
                )
                placeholder_time = max(
                    timestamp
                    for timestamp in (
                        state.operation.updated_at,
                        task.updated_at,
                        active_lease.acquired_at,
                        active_lease.expires_at,
                        active_lease.started_at,
                        active_lease.renewed_at,
                    )
                    if timestamp is not None
                )
                lease_index, _ = self._lease(state, guard)
                state.task_leases[lease_index] = replace(
                    active_lease,
                    released_at=placeholder_time,
                    release_reason="recovery_requested",
                )
                state.operation = replace(
                    state.operation,
                    updated_at=placeholder_time,
                )
                model_call = self._completed_model_call_for_turn(
                    state,
                    task.turn_id,
                )
                state.events.append(
                    RuntimeEvent(
                        id=self._id_factory("event"),
                        type="task.lease_lost",
                        agent_id=state.operation.agent_id,
                        operation_id=operation_id,
                        session_id=state.operation.session_id,
                        turn_id=task.turn_id,
                        model_call_id=model_call.id,
                        call_id=task.call_id,
                        task_id=task.id,
                        capability_id=task.capability_id,
                        executor_id=task.executor_id,
                        payload={},
                        created_at=placeholder_time,
                    )
                )
                try:
                    recovered, cancellation_requested = await _await_store_write(
                        self._store.recover_expired_task(
                            self._snapshot(state),
                            expected_revision=state.revision,
                            guard=guard,
                        )
                    )
                except OperationStoreError as error:
                    raise OperationStateError(
                        f"task recovery rejected: {task.id}"
                    ) from error
                if cancellation_requested:
                    raise asyncio.CancelledError
                state = self._state_from_snapshot(
                    recovered.operation.snapshot,
                    revision=recovered.operation.revision,
                )
                _, task = self._task(state, task.id)
                if task.status in {
                    TaskStatus.CANCELLED,
                    TaskStatus.MANUAL_RECOVERY_REQUIRED,
                }:
                    return None
                if task.status is not TaskStatus.READY:
                    raise OperationStateError(
                        f"task recovery produced non-executable state: "
                        f"{task.status.value}"
                    )
            elif task.status is TaskStatus.PENDING:
                needs_governance = True
            elif task.status is not TaskStatus.READY:
                raise OperationStateError(f"task is not resumable: {task.status.value}")

        if needs_governance:
            decision = await self._govern_task(operation_id, task_id)
            if decision.effect is not PolicyEffect.ALLOW:
                return None
        return await self._execute_materialized_task(
            operation_id,
            task_id,
            requested_timeout_seconds=requested_timeout_seconds,
        )

    async def _materialize_task(self, proposal: ActionProposal) -> Task:
        """Persist one exact unexecuted task before any executor I/O."""

        # Commit the unexecuted task first. The proposal is untrusted until it
        # is bound to an exact tool call in a committed model response.
        async with self._lock:
            state = await self._working_state(proposal.operation_id)
            model_call, tool_call = self._committed_tool_call(
                state,
                proposal.turn_id,
                proposal.call_id,
            )
            self._require_preceding_calls_observed(state, model_call, tool_call)
            try:
                view, capability = self._capabilities.resolve_tool(tool_call.name)
            except KeyError as error:
                raise OperationStateError(
                    f"model selected an unknown tool view: {tool_call.name}"
                ) from error
            expected_definition = self._capabilities.tool_definition(view.name)
            if expected_definition not in model_call.request.tools:
                raise OperationStateError(
                    f"model selected a tool without its declared projection: "
                    f"{view.name}"
                )
            if proposal.capability_id != view.capability_id:
                raise OperationStateError(
                    "proposal capability does not match the committed tool view"
                )
            if proposal.arguments != tool_call.arguments:
                raise OperationStateError(
                    "proposal arguments do not match the committed tool call"
                )
            validated_arguments = self._capabilities.validate_arguments(
                capability.id,
                tool_call.arguments,
            )
            self._capabilities.resolve_execution(capability.id)
            if any(
                task.turn_id == proposal.turn_id and task.call_id == proposal.call_id
                for task in state.tasks
            ):
                raise OperationStateError(
                    f"tool call already materialized: {proposal.call_id}"
                )
            now = self._clock()
            task_id = self._id_factory("task")
            idempotency_key = (
                f"{proposal.operation_id}:{task_id}"
                if capability.side_effecting and capability.idempotent
                else None
            )
            task = Task(
                id=task_id,
                operation_id=proposal.operation_id,
                turn_id=proposal.turn_id,
                call_id=proposal.call_id,
                capability_id=capability.id,
                executor_id=capability.executor_id,
                status=TaskStatus.PENDING,
                attempt=1,
                arguments=validated_arguments,
                created_at=now,
                updated_at=now,
                execution_facts=TaskExecutionFacts(
                    capability_fingerprint=capability.contract_fingerprint,
                    arguments_hash=(
                        "sha256:"
                        + hashlib.sha256(
                            canonical_json(validated_arguments).encode("utf-8")
                        ).hexdigest()
                    ),
                    access_mode=capability.access_mode,
                    risk=capability.risk,
                    side_effecting=capability.side_effecting,
                    idempotent=capability.idempotent,
                    replay_safe=capability.replay_safe,
                    idempotency_key=idempotency_key,
                ),
            )
            state.tasks.append(task)
            state.loop_state = replace(
                state.loop_state,
                phase=LoopPhase.AWAITING_EXECUTION,
                action_count=state.loop_state.action_count + 1,
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "task.created",
                turn_id=task.turn_id,
                model_call_id=model_call.id,
                call_id=task.call_id,
                task_id=task.id,
                capability_id=task.capability_id,
                executor_id=task.executor_id,
                payload={
                    "task_id": task.id,
                    "capability_id": task.capability_id,
                    "executor_id": task.executor_id,
                },
            )
            await self._commit(state)

        return task

    async def _execute_materialized_task(
        self,
        operation_id: str,
        task_id: str,
        *,
        requested_timeout_seconds: float | None,
    ) -> Evidence:

        # Persist readiness, claim a fenced lease, and durably cross the
        # executor-start boundary before the only external execution call.
        post_start_wall_deadline_error: OperationWallTimeExceeded | None = None
        lease_expired_before_io = False
        execution_timeout_seconds = 0.0
        timeout_is_wall = False
        executor: Executor | None = None
        guard: TaskLeaseGuard | None = None
        async with self._lock:
            state = await self._working_state(operation_id)
            _, committed_task = self._task(state, task_id)
            task = committed_task
            model_call = self._completed_model_call_for_turn(state, task.turn_id)
            if committed_task.status is not TaskStatus.READY:
                raise OperationStateError("task is no longer executable")
            self._resolve_task_execution(committed_task)
            ready_task = committed_task

            claim_event = RuntimeEvent(
                id=self._id_factory("event"),
                type="task.claimed",
                agent_id=state.operation.agent_id,
                operation_id=state.operation.id,
                session_id=state.operation.session_id,
                turn_id=ready_task.turn_id,
                model_call_id=model_call.id,
                call_id=ready_task.call_id,
                task_id=ready_task.id,
                capability_id=ready_task.capability_id,
                executor_id=ready_task.executor_id,
                payload={},
                created_at=self._clock(),
            )
            try:
                claim, cancellation_requested = await _await_store_write(
                    self._store.claim_task(
                        TaskClaimRequest(
                            operation_id=ready_task.operation_id,
                            task_id=ready_task.id,
                            holder_id=self._lease_holder_id,
                            lease_duration_seconds=self._lease_duration_seconds,
                            event=claim_event,
                        ),
                        expected_revision=state.revision,
                    )
                )
            except OperationStoreError as error:
                raise OperationStateError(
                    f"task claim rejected: {ready_task.id}"
                ) from error
            if cancellation_requested:
                raise asyncio.CancelledError

            state = self._state_from_snapshot(
                claim.commit_result.operation.snapshot,
                revision=claim.commit_result.operation.revision,
            )
            task_index, claimed_task = self._task(state, ready_task.id)
            _, executor = self._resolve_task_execution(claimed_task)
            guard = TaskLeaseGuard(
                operation_id=claimed_task.operation_id,
                task_id=claimed_task.id,
                holder_id=claim.lease.holder_id,
                attempt=claim.lease.attempt,
                fencing_token=claim.lease.fencing_token,
            )
            lease_index, claimed_lease = self._lease(state, guard)
            state.tasks[task_index] = replace(
                claimed_task,
                status=TaskStatus.RUNNING,
                updated_at=claimed_task.updated_at,
            )
            state.task_leases[lease_index] = replace(
                claimed_lease,
                started_at=claimed_lease.acquired_at,
            )
            self._append_event(
                state,
                "executor.started",
                turn_id=claimed_task.turn_id,
                model_call_id=model_call.id,
                call_id=claimed_task.call_id,
                task_id=claimed_task.id,
                capability_id=claimed_task.capability_id,
                executor_id=claimed_task.executor_id,
                payload={
                    "task_id": claimed_task.id,
                    "executor_id": claimed_task.executor_id,
                },
            )
            fenced_start_monotonic = asyncio.get_running_loop().time()
            state = await self._commit_fenced(state, guard)
            fenced_start_round_trip = max(
                0.0,
                asyncio.get_running_loop().time() - fenced_start_monotonic,
            )
            _, task = self._task(state, claimed_task.id)
            _, started_lease = self._lease(state, guard)
            _, executor = self._resolve_task_execution(task)
            if started_lease.started_at is None:
                raise OperationStateError("executor start lease was not committed")

            # Readiness, claim, and fenced start are durable writes and may
            # consume the remaining operation budget. Recompute both the
            # wall and lease bounds from the committed start immediately
            # before allowing executor I/O.
            execution_now = self._clock()
            if execution_now.tzinfo is None or execution_now.utcoffset() is None:
                raise ValueError("runtime clock must return a timezone-aware datetime")
            elapsed_seconds = max(
                0.0,
                (execution_now - state.operation.created_at).total_seconds(),
            )
            remaining_wall_time = state.budgets.max_wall_time_seconds - elapsed_seconds
            # Lease timestamps belong to the store's authoritative clock;
            # never compare them with the independently injectable runtime
            # clock. The fenced start supplies the exact remaining store
            # interval at that commit boundary.
            remaining_lease_time = (
                started_lease.expires_at - started_lease.started_at
            ).total_seconds() - fenced_start_round_trip
            if remaining_wall_time <= 0:
                post_start_wall_deadline_error = OperationWallTimeExceeded(task.id)
            elif remaining_lease_time <= 0:
                lease_expired_before_io = True
            else:
                # Reserve half of the live interval for cancellation and a
                # fenced outcome. Until P2-05f adds renewal, the other half
                # is the strict executor-timeout ceiling.
                lease_safe_timeout = remaining_lease_time / 2.0
                non_wall_timeout = min(
                    state.budgets.task_timeout_seconds,
                    (
                        requested_timeout_seconds
                        if requested_timeout_seconds is not None
                        else math.inf
                    ),
                    lease_safe_timeout,
                )
                execution_timeout_seconds = min(
                    remaining_wall_time,
                    non_wall_timeout,
                )
                timeout_is_wall = remaining_wall_time <= non_wall_timeout

        assert executor is not None
        assert guard is not None
        if post_start_wall_deadline_error is not None:
            await self._fail_task(
                task.operation_id,
                task.id,
                guard,
                "task_timeout",
            )
            raise post_start_wall_deadline_error
        if lease_expired_before_io:
            raise OperationStateError(
                f"task lease expired before executor invocation: {task.id}"
            )
        assert execution_timeout_seconds > 0

        request = ExecutionRequest(
            operation_id=task.operation_id,
            task_id=task.id,
            turn_id=task.turn_id,
            capability_id=task.capability_id,
            executor_id=task.executor_id,
            attempt=task.attempt,
            fencing_token=guard.fencing_token,
            idempotency_key=task.execution_facts.idempotency_key,
            arguments=task.arguments,
        )
        timeout = asyncio.timeout(execution_timeout_seconds)
        deadline_cause: Exception | None = None
        try:
            async with timeout:
                candidate = await executor.execute(request)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise asyncio.CancelledError from error
            if timeout.expired():
                deadline_cause = error
            else:
                await self._record_execution_failure(
                    task.operation_id,
                    task.id,
                    guard,
                    "executor_failed",
                )
                raise CapabilityExecutionError(
                    f"executor failed for task {task.id}"
                ) from error

        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError
        if timeout.expired():
            await self._record_execution_failure(
                task.operation_id,
                task.id,
                guard,
                "task_timeout",
            )
            if timeout_is_wall:
                raise OperationWallTimeExceeded(task.id) from deadline_cause
            raise TaskExecutionTimeout(
                task.id,
                execution_timeout_seconds,
            ) from deadline_cause

        if deadline_cause is not None:
            # ``timeout.expired()`` is stable once the context has exited, so
            # this branch is defensive against an invalid timeout lifecycle.
            await self._record_execution_failure(
                task.operation_id,
                task.id,
                guard,
                "executor_failed",
            )
            raise CapabilityExecutionError(
                f"executor failed for task {task.id}"
            ) from deadline_cause

        execution_identity_error: Exception | None = None
        accepted_candidate: EvidenceCandidate | None = None
        validation_error: EvidenceValidationError | None = None
        evidence_id: str | None = None
        blob_request: BlobPut | None = None
        async with self._lock:
            state = await self._working_state(task.operation_id)
            _, committed_task = self._task(state, task.id)
            if committed_task.status is not TaskStatus.RUNNING:
                raise OperationStateError("task is no longer running")
            self._lease(state, guard)
            try:
                _, current_executor = self._resolve_task_execution(committed_task)
            except OperationStateError as error:
                execution_identity_error = error
            else:
                if current_executor is not executor:
                    execution_identity_error = OperationStateError(
                        "task execution identity changed"
                    )

            if execution_identity_error is None:
                try:
                    accepted_candidate = self._capabilities.validate_evidence(
                        committed_task.capability_id,
                        candidate,
                    )
                except EvidenceValidationError as error:
                    # Failure publication needs its own copy-on-write transition.
                    validation_error = error

            if accepted_candidate is not None:
                evidence_id = self._id_factory("evidence")
                artifact = accepted_candidate.artifact
                if artifact is not None:
                    blob_id = self._id_factory("blob")
                    digest = "sha256:" + hashlib.sha256(artifact.content).hexdigest()
                    blob_request = BlobPut(
                        blob_id=blob_id,
                        media_type=artifact.media_type,
                        created_at=self._clock(),
                        sensitivity_class=artifact.sensitivity_class,
                        retention_class=artifact.retention_class,
                        operation_id=task.operation_id,
                        task_id=task.id,
                        evidence_id=evidence_id,
                        expected_digest=digest,
                        encryption_metadata=artifact.encryption_metadata,
                    )

        if execution_identity_error is not None:
            await self._record_execution_failure(
                task.operation_id,
                task.id,
                guard,
                "execution_identity_changed",
            )
            raise CapabilityExecutionError(
                f"executor identity changed for task {task.id}"
            ) from execution_identity_error
        if validation_error is not None:
            await self._record_execution_failure(
                task.operation_id,
                task.id,
                guard,
                "evidence_rejected",
            )
            raise validation_error

        assert accepted_candidate is not None
        assert evidence_id is not None
        blob_metadata: BlobMetadata | None = None
        if blob_request is not None:
            artifact = accepted_candidate.artifact
            assert artifact is not None
            try:
                if self._blob_store is None:
                    raise RuntimeError(
                        "artifact evidence requires a configured BlobStore"
                    )
                blob_metadata = await self._blob_store.put(
                    blob_request,
                    artifact.content,
                )
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise asyncio.CancelledError
                expected_metadata = BlobMetadata(
                    blob_id=blob_request.blob_id,
                    digest=blob_request.expected_digest or "",
                    size_bytes=len(artifact.content),
                    media_type=blob_request.media_type,
                    created_at=blob_request.created_at,
                    sensitivity_class=blob_request.sensitivity_class,
                    retention_class=blob_request.retention_class,
                    operation_id=blob_request.operation_id,
                    task_id=blob_request.task_id,
                    evidence_id=blob_request.evidence_id,
                    encryption_metadata=blob_request.encryption_metadata,
                )
                if (
                    not isinstance(blob_metadata, BlobMetadata)
                    or blob_metadata != expected_metadata
                ):
                    raise RuntimeError(
                        "blob store returned metadata that differs from the "
                        "requested evidence provenance"
                    )
            except asyncio.CancelledError:
                raise
            except Exception as error:
                current_task = asyncio.current_task()
                if current_task is not None and current_task.cancelling():
                    raise asyncio.CancelledError from error
                await self._record_execution_failure(
                    task.operation_id,
                    task.id,
                    guard,
                    "evidence_blob_failed",
                )
                raise CapabilityExecutionError(
                    f"artifact persistence failed for task {task.id}"
                ) from error

        async with self._lock:
            state = await self._working_state(task.operation_id)
            task_index, committed_task = self._task(state, task.id)
            model_call = self._completed_model_call_for_turn(state, task.turn_id)
            if committed_task.status is not TaskStatus.RUNNING:
                raise OperationStateError("task is no longer running")
            try:
                _, current_executor = self._resolve_task_execution(committed_task)
            except OperationStateError as error:
                execution_identity_error = error
            else:
                if current_executor is not executor:
                    execution_identity_error = OperationStateError(
                        "task execution identity changed"
                    )

            if execution_identity_error is None:
                placeholder_time = committed_task.updated_at
                payload_json = canonical_json(accepted_candidate.payload)
                evidence = Evidence(
                    id=evidence_id,
                    operation_id=task.operation_id,
                    task_id=task.id,
                    turn_id=task.turn_id,
                    capability_id=task.capability_id,
                    executor_id=task.executor_id,
                    kind=accepted_candidate.kind,
                    schema_version=accepted_candidate.schema_version,
                    attempt=committed_task.attempt,
                    accepted=True,
                    payload=accepted_candidate.payload,
                    content_hash=(
                        blob_metadata.digest
                        if blob_metadata is not None
                        else "sha256:"
                        + hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
                    ),
                    created_at=placeholder_time,
                    blob_id=(None if blob_metadata is None else blob_metadata.blob_id),
                )
                state.evidence.append(evidence)
                state.tasks[task_index] = replace(
                    committed_task,
                    status=TaskStatus.SUCCEEDED,
                    evidence_ids=(evidence.id,),
                    updated_at=placeholder_time,
                )
                lease_index, active_lease = self._lease(state, guard)
                release_placeholder = max(
                    timestamp
                    for timestamp in (
                        active_lease.acquired_at,
                        active_lease.started_at,
                        active_lease.renewed_at,
                    )
                    if timestamp is not None
                )
                state.task_leases[lease_index] = replace(
                    active_lease,
                    released_at=release_placeholder,
                    release_reason="completed",
                )
                self._append_event(
                    state,
                    "executor.completed",
                    turn_id=task.turn_id,
                    model_call_id=model_call.id,
                    call_id=task.call_id,
                    task_id=task.id,
                    capability_id=task.capability_id,
                    executor_id=task.executor_id,
                    payload={"task_id": task.id, "executor_id": task.executor_id},
                )
                self._append_event(
                    state,
                    "evidence.accepted",
                    turn_id=task.turn_id,
                    model_call_id=model_call.id,
                    call_id=task.call_id,
                    task_id=task.id,
                    evidence_id=evidence.id,
                    capability_id=task.capability_id,
                    executor_id=task.executor_id,
                    payload={"task_id": task.id, "evidence_id": evidence.id},
                )
                self._append_event(
                    state,
                    "task.succeeded",
                    turn_id=task.turn_id,
                    model_call_id=model_call.id,
                    call_id=task.call_id,
                    task_id=task.id,
                    evidence_id=evidence.id,
                    capability_id=task.capability_id,
                    executor_id=task.executor_id,
                    payload={"task_id": task.id},
                )
                committed_state = await self._commit_fenced(state, guard)
                return self._accepted_evidence(committed_state, evidence.id)

        if execution_identity_error is not None:
            await self._record_execution_failure(
                task.operation_id,
                task.id,
                guard,
                "execution_identity_changed",
            )
            raise CapabilityExecutionError(
                f"executor identity changed for task {task.id}"
            ) from execution_identity_error
        raise AssertionError("validated evidence did not reach a terminal outcome")

    async def append_observation(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        async with self._lock:
            state = await self._working_state(observation.operation_id)
            if observation.evidence_id is None or observation.task_id is None:
                raise OperationStateError(
                    "executor observation requires task and evidence identity"
                )
            _, task = self._task(state, observation.task_id)
            model_call = self._completed_model_call_for_turn(
                state,
                observation.turn_id,
            )
            evidence = self._accepted_evidence(state, observation.evidence_id)
            if task.status is not TaskStatus.SUCCEEDED:
                raise OperationStateError("observation task is not succeeded")
            if evidence.id not in task.evidence_ids:
                raise OperationStateError("observation evidence is not linked to task")
            if observation.call_id is not None and observation.call_id != task.call_id:
                raise OperationStateError("observation call_id does not match its task")
            if evidence.task_id != task.id or evidence.turn_id != observation.turn_id:
                raise OperationStateError(
                    "observation linkage does not match accepted evidence"
                )
            if any(
                item.evidence_id == observation.evidence_id
                for item in state.observations
            ):
                raise OperationStateError("evidence already has an observation")
            state.observations.append(observation)
            now = self._clock()
            state.loop_state = replace(
                state.loop_state,
                phase=LoopPhase.OBSERVING,
                identical_failure_count=0,
                no_progress_fingerprints=(),
                observation_characters=(
                    state.loop_state.observation_characters
                    + self._observation_characters(observation)
                ),
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "observation.recorded",
                turn_id=observation.turn_id,
                model_call_id=model_call.id,
                call_id=task.call_id,
                task_id=task.id,
                evidence_id=evidence.id,
                capability_id=task.capability_id,
                executor_id=task.executor_id,
                payload={
                    "task_id": task.id,
                    "evidence_id": evidence.id,
                    "code": observation.code,
                    "truncated": observation.truncated,
                },
            )
            await self._commit(state)

    async def complete(self, operation_id: str, final_text: str) -> LoopExit:
        async with self._lock:
            _required_text(final_text, "final_text")
            state = await self._working_state(operation_id)
            if not state.readiness or not state.readiness[-1].allowed:
                raise OperationStateError("completion requires allowed readiness")
            if state.loop_state.final_answer_candidate != final_text:
                raise OperationStateError(
                    "completion text differs from readiness candidate"
                )
            now = self._clock()
            state.operation = replace(
                state.operation,
                status=OperationStatus.SUCCEEDED,
                updated_at=now,
                final_text=final_text,
                terminal_reason="completed",
            )
            state.loop_state = replace(state.loop_state, phase=LoopPhase.TERMINAL)
            self._append_event(
                state,
                "operation.succeeded",
                payload={"reason": "completed"},
            )
            result = LoopExit(
                operation_id=operation_id,
                kind=LoopExitKind.COMPLETED,
                reason="completed",
                final_text=final_text,
                created_at=now,
            )
            await self._commit(state)
            return result

    async def fail(
        self,
        operation_id: str,
        reason: str,
        *,
        final_text: str | None = None,
    ) -> LoopExit:
        async with self._lock:
            _required_text(reason, "failure reason")
            state = await self._working_state(operation_id)
            result = self._fail_locked(state, reason, final_text=final_text)
            await self._commit(state)
            return result

    async def fail_no_progress(
        self,
        operation_id: str,
        call_id: str,
    ) -> LoopExit:
        _required_text(call_id, "no-progress call_id")
        async with self._lock:
            state = await self._working_state(operation_id)
            if not state.turns:
                raise OperationStateError("no-progress failure requires a turn")
            turn_id = state.turns[-1].id
            model_call, committed_call = self._committed_tool_call(
                state,
                turn_id,
                call_id,
            )
            fingerprint = self._action_fingerprint(committed_call)
            rejection_observation = next(
                (
                    observation
                    for observation in state.observations
                    if observation.turn_id == turn_id
                    and observation.call_id == call_id
                    and not observation.success
                    and observation.task_id is None
                    and observation.evidence_id is None
                ),
                None,
            )
            rejection_event = next(
                (
                    event
                    for event in state.events
                    if event.type == "action.rejected"
                    and event.turn_id == turn_id
                    and event.call_id == call_id
                ),
                None,
            )
            if rejection_observation is None or rejection_event is None:
                raise OperationStateError(
                    "no-progress failure requires a committed current rejection"
                )
            if rejection_event.payload.get("fingerprint") != fingerprint:
                raise OperationStateError(
                    "current rejection fingerprint does not match its tool call"
                )
            if (
                not state.loop_state.no_progress_fingerprints
                or state.loop_state.no_progress_fingerprints[-1] != fingerprint
            ):
                raise OperationStateError(
                    "no-progress fingerprint does not match committed loop state"
                )
            reason = "no_progress_action_failure_limit"
            self._append_event(
                state,
                "no_progress.detected",
                turn_id=turn_id,
                model_call_id=model_call.id,
                call_id=call_id,
                payload={
                    "count": state.loop_state.identical_failure_count,
                    "fingerprint": fingerprint,
                    "reason": reason,
                },
            )
            result = self._fail_locked(state, reason)
            await self._commit(state)
            return result

    async def fail_budget(
        self,
        operation_id: str,
        reason: str,
        *,
        budget: str,
        limit: int | float | str,
        used: int | float | str,
        turn_id: str | None = None,
        call_id: str | None = None,
        task_id: str | None = None,
    ) -> LoopExit:
        """Atomically persist one loop-owned budget decision and failure."""

        _required_text(reason, "budget failure reason")
        _required_text(budget, "budget kind")
        async with self._lock:
            state = await self._working_state(operation_id)
            if state.model_calls and (
                state.model_calls[-1].status is ModelCallStatus.STARTED
            ):
                model_call = state.model_calls[-1]
                state.model_calls[-1] = replace(
                    model_call,
                    status=ModelCallStatus.FAILED,
                    error_code=reason,
                    updated_at=self._clock(),
                )
                self._append_event(
                    state,
                    "model_call.failed",
                    turn_id=model_call.turn_id,
                    model_call_id=model_call.id,
                    payload={
                        "error_code": reason,
                        "model_call_id": model_call.id,
                    },
                )
            self._append_event(
                state,
                "budget.exhausted",
                turn_id=turn_id,
                model_call_id=next(
                    (
                        model_call.id
                        for model_call in reversed(state.model_calls)
                        if model_call.turn_id == turn_id
                    ),
                    None,
                ),
                call_id=call_id,
                task_id=task_id,
                payload={
                    "budget": budget,
                    "limit": limit,
                    "reason": reason,
                    "used": used,
                },
            )
            result = self._fail_locked(state, reason)
            await self._commit(state)
            return result

    async def interrupt(
        self,
        operation_id: str,
        reason: str = "run_cancelled",
    ) -> LoopExit:
        """Persist interruption and active-child cancellation intent atomically."""

        _required_text(reason, "interruption reason")
        async with self._lock:
            while True:
                state = await self._working_state(operation_id)
                result = self._interrupt_locked(state, reason)
                try:
                    await self._commit(state)
                except OperationStateError as error:
                    if isinstance(error.__cause__, OperationRevisionConflict):
                        continue
                    raise
                return result

    async def record_model_failure(
        self,
        operation_id: str,
        model_call_id: str,
        error_code: str,
    ) -> LoopExit:
        async with self._lock:
            _required_text(error_code, "model error_code")
            state = await self._working_state(operation_id)
            call_index, model_call = self._model_call(state, model_call_id)
            if model_call.status is not ModelCallStatus.STARTED:
                raise OperationStateError("model call is already terminal")
            now = self._clock()
            state.model_calls[call_index] = replace(
                model_call,
                status=ModelCallStatus.FAILED,
                error_code=error_code,
                updated_at=now,
            )
            self._append_event(
                state,
                "model_call.failed",
                turn_id=model_call.turn_id,
                model_call_id=model_call.id,
                payload={"error_code": error_code, "model_call_id": model_call.id},
            )
            result = self._fail_locked(state, error_code)
            await self._commit(state)
            return result

    async def inspect(self, operation_id: str) -> OperationSnapshot:
        async with self._lock:
            try:
                committed = await self._store.load(operation_id)
            except OperationNotFoundError as error:
                raise KeyError(f"Unknown operation: {operation_id}") from error
            return committed.snapshot

    async def inspect_nonterminal(
        self,
        agent_id: str,
    ) -> tuple[OperationSnapshot, ...]:
        """Inspect one agent's resumable operations without exposing revisions."""

        async with self._lock:
            committed = await self._store.load_nonterminal(agent_id)
            return tuple(item.snapshot for item in committed)

    async def elapsed_seconds(self, operation_id: str) -> float:
        """Return authoritative elapsed wall time from the runtime clock."""

        async with self._lock:
            try:
                committed = await self._store.load(operation_id)
            except OperationNotFoundError as error:
                raise KeyError(f"Unknown operation: {operation_id}") from error
            now = self._clock()
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError("runtime clock must return a timezone-aware datetime")
            return max(
                0.0,
                (now - committed.snapshot.operation.created_at).total_seconds(),
            )

    async def _working_state(self, operation_id: str) -> _OperationState:
        try:
            committed = await self._store.load(operation_id)
        except OperationNotFoundError as error:
            raise KeyError(f"Unknown operation: {operation_id}") from error
        snapshot = committed.snapshot
        if snapshot.operation.status in {
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
            OperationStatus.INTERRUPTED,
        }:
            raise OperationStateError("operation is already terminal")
        return self._state_from_snapshot(snapshot, revision=committed.revision)

    @staticmethod
    def _state_from_snapshot(
        snapshot: OperationSnapshot,
        *,
        revision: int,
    ) -> _OperationState:
        return _OperationState(
            revision=revision,
            trigger=snapshot.trigger,
            operation=snapshot.operation,
            loop_state=snapshot.loop_state,
            budgets=snapshot.budgets,
            turns=list(snapshot.turns),
            model_calls=list(snapshot.model_calls),
            readiness=list(snapshot.readiness),
            tasks=list(snapshot.tasks),
            task_dependencies=list(snapshot.task_dependencies),
            task_leases=list(snapshot.task_leases),
            approvals=list(snapshot.approvals),
            evidence=list(snapshot.evidence),
            observations=list(snapshot.observations),
            events=list(snapshot.events),
        )

    async def _commit(self, state: _OperationState) -> None:
        try:
            committed, cancellation_requested = await _await_store_write(
                self._store.commit(
                    self._snapshot(state),
                    expected_revision=state.revision,
                )
            )
        except OperationNotFoundError as error:
            raise KeyError(f"Unknown operation: {state.operation.id}") from error
        except OperationRevisionConflict as error:
            raise OperationStateError(
                f"operation changed concurrently: {state.operation.id}"
            ) from error
        except InvalidOperationCheckpointError as error:
            raise OperationStateError(
                f"operation checkpoint rejected: {state.operation.id}"
            ) from error
        state.revision = committed.operation.revision
        if cancellation_requested:
            raise asyncio.CancelledError

    async def _commit_fenced(
        self,
        state: _OperationState,
        guard: TaskLeaseGuard,
    ) -> _OperationState:
        try:
            committed, cancellation_requested = await _await_store_write(
                self._store.commit_fenced(
                    self._snapshot(state),
                    expected_revision=state.revision,
                    guard=guard,
                )
            )
        except OperationNotFoundError as error:
            raise KeyError(f"Unknown operation: {state.operation.id}") from error
        except OperationStoreError as error:
            raise OperationStateError(
                f"fenced task checkpoint rejected: {guard.task_id}"
            ) from error
        committed_state = self._state_from_snapshot(
            committed.operation.snapshot,
            revision=committed.operation.revision,
        )
        if cancellation_requested:
            raise asyncio.CancelledError
        return committed_state

    def _turn(self, state: _OperationState, turn_id: str) -> tuple[int, Turn]:
        for index, turn in enumerate(state.turns):
            if turn.id == turn_id:
                return index, turn
        raise OperationStateError(f"unknown turn: {turn_id}")

    def _model_call(
        self,
        state: _OperationState,
        model_call_id: str,
    ) -> tuple[int, ModelCall]:
        for index, model_call in enumerate(state.model_calls):
            if model_call.id == model_call_id:
                return index, model_call
        raise OperationStateError(f"unknown model call: {model_call_id}")

    @staticmethod
    def _action_fingerprint(call: ToolCall) -> str:
        normalized = canonical_json(
            {
                "arguments": call.arguments,
                "tool_name": call.name,
            }
        )
        return "sha256:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def _completed_model_call_for_turn(
        self,
        state: _OperationState,
        turn_id: str,
    ) -> ModelCall:
        self._turn(state, turn_id)
        for model_call in state.model_calls:
            if model_call.turn_id != turn_id:
                continue
            if (
                model_call.status is ModelCallStatus.COMPLETED
                and model_call.response is not None
            ):
                return model_call
            raise OperationStateError("proposal requires a completed model response")
        raise OperationStateError("proposal turn has no committed model response")

    def _committed_tool_call(
        self,
        state: _OperationState,
        turn_id: str,
        call_id: str,
    ) -> tuple[ModelCall, ToolCall]:
        if not state.turns or state.turns[-1].id != turn_id:
            raise OperationStateError("action does not belong to the current turn")
        if state.loop_state.phase not in {
            LoopPhase.VALIDATING_ACTION,
            LoopPhase.OBSERVING,
        }:
            raise OperationStateError("operation is not accepting an action")
        model_call = self._completed_model_call_for_turn(state, turn_id)
        assert model_call.response is not None
        tool_call = next(
            (call for call in model_call.response.tool_calls if call.id == call_id),
            None,
        )
        if tool_call is None:
            raise OperationStateError(
                f"action tool call is not in the committed response: {call_id}"
            )
        return model_call, tool_call

    @staticmethod
    def _require_preceding_calls_observed(
        state: _OperationState,
        model_call: ModelCall,
        tool_call: ToolCall,
    ) -> None:
        assert model_call.response is not None
        call_position = model_call.response.tool_calls.index(tool_call)
        for earlier_call in model_call.response.tool_calls[:call_position]:
            earlier_task = next(
                (
                    task
                    for task in state.tasks
                    if task.turn_id == model_call.turn_id
                    and task.call_id == earlier_call.id
                ),
                None,
            )
            if (
                earlier_task is None
                or earlier_task.status is not TaskStatus.SUCCEEDED
                or not earlier_task.evidence_ids
                or any(
                    not any(
                        observation.evidence_id == evidence_id
                        for observation in state.observations
                    )
                    for evidence_id in earlier_task.evidence_ids
                )
            ):
                raise OperationStateError(
                    "action tool call is out of committed sequential order"
                )

    @staticmethod
    def _observation_characters(observation: Observation) -> int:
        return len(observation.message) + len(canonical_json(observation.payload))

    def _task(self, state: _OperationState, task_id: str) -> tuple[int, Task]:
        for index, task in enumerate(state.tasks):
            if task.id == task_id:
                return index, task
        raise OperationStateError(f"unknown task: {task_id}")

    @staticmethod
    def _approval(
        state: _OperationState,
        approval_id: str,
    ) -> tuple[int, ApprovalRequest]:
        for index, approval in enumerate(state.approvals):
            if approval.id == approval_id:
                return index, approval
        raise OperationStateError(f"unknown approval: {approval_id}")

    @staticmethod
    def _governance_facts(
        state: _OperationState,
        task: Task,
    ) -> GovernanceFacts:
        facts = task.execution_facts
        return GovernanceFacts(
            operation_id=task.operation_id,
            task_id=task.id,
            capability_id=task.capability_id,
            executor_id=task.executor_id,
            capability_fingerprint=facts.capability_fingerprint,
            arguments_hash=facts.arguments_hash,
            access_mode=facts.access_mode,
            risk=facts.risk,
            side_effecting=facts.side_effecting,
            idempotent=facts.idempotent,
            replay_safe=facts.replay_safe,
            idempotency_key=facts.idempotency_key,
            validation_passed=True,
            in_scope=True,
            destructive=False,
            sensitivity_class="internal",
            actor_id=f"{state.trigger.kind.value}:{state.trigger.source_id}",
        )

    @staticmethod
    def _lease(
        state: _OperationState,
        guard: TaskLeaseGuard,
    ) -> tuple[int, TaskLease]:
        for index, lease in enumerate(state.task_leases):
            if (
                lease.operation_id == guard.operation_id
                and lease.task_id == guard.task_id
                and lease.holder_id == guard.holder_id
                and lease.attempt == guard.attempt
                and lease.fencing_token == guard.fencing_token
            ):
                return index, lease
        raise OperationStateError(f"unknown task lease: {guard.task_id}")

    def _resolve_task_execution(self, task: Task) -> tuple[Capability, Executor]:
        try:
            capability, executor = self._capabilities.resolve_execution(
                task.capability_id
            )
        except (KeyError, ValueError) as error:
            raise OperationStateError(
                f"task execution identity changed: {task.id}"
            ) from error
        expected_idempotency_key = (
            f"{task.operation_id}:{task.id}"
            if capability.side_effecting and capability.idempotent
            else None
        )
        expected_facts = TaskExecutionFacts(
            capability_fingerprint=capability.contract_fingerprint,
            arguments_hash=(
                "sha256:"
                + hashlib.sha256(
                    canonical_json(task.arguments).encode("utf-8")
                ).hexdigest()
            ),
            access_mode=capability.access_mode,
            risk=capability.risk,
            side_effecting=capability.side_effecting,
            idempotent=capability.idempotent,
            replay_safe=capability.replay_safe,
            idempotency_key=expected_idempotency_key,
        )
        if (
            task.executor_id != capability.executor_id
            or task.execution_facts != expected_facts
        ):
            raise OperationStateError(
                f"task persisted execution facts changed: {task.id}"
            )
        return capability, executor

    @staticmethod
    def _accepted_evidence(state: _OperationState, evidence_id: str) -> Evidence:
        for evidence in state.evidence:
            if evidence.id == evidence_id and evidence.accepted:
                return evidence
        raise OperationStateError(f"unknown accepted evidence: {evidence_id}")

    async def _record_execution_failure(
        self,
        operation_id: str,
        task_id: str,
        guard: TaskLeaseGuard,
        error_code: str,
    ) -> None:
        """Fail safe I/O, but retain side-effect uncertainty for recovery."""

        fail_known_safe = False
        async with self._lock:
            state = await self._working_state(operation_id)
            task_index, task = self._task(state, task_id)
            if task.status is not TaskStatus.RUNNING:
                raise OperationStateError("task is no longer running")
            _, active_lease = self._lease(state, guard)
            if active_lease.released_at is not None or task.attempt != guard.attempt:
                raise OperationStateError("task execution lease is no longer active")
            if not task.execution_facts.side_effecting:
                fail_known_safe = True
            elif task.cancellation_requested:
                return
            else:
                fail_known_safe = False
                now = self._clock()
                if now.tzinfo is None or now.utcoffset() is None:
                    raise ValueError(
                        "runtime clock must return a timezone-aware datetime"
                    )
                now = max(now, task.updated_at, state.operation.updated_at)
                state.tasks[task_index] = replace(
                    task,
                    updated_at=now,
                )
                state.operation = replace(state.operation, updated_at=now)
                model_call = self._completed_model_call_for_turn(
                    state,
                    task.turn_id,
                )
                self._append_event(
                    state,
                    "task.outcome_unknown",
                    turn_id=task.turn_id,
                    model_call_id=model_call.id,
                    call_id=task.call_id,
                    task_id=task.id,
                    capability_id=task.capability_id,
                    executor_id=task.executor_id,
                    payload={
                        "reason": error_code,
                        "status": task.status.value,
                        "attempt": guard.attempt,
                        "fencing_token": guard.fencing_token,
                    },
                )
                try:
                    await self._commit_fenced(state, guard)
                except OperationStateError as error:
                    if isinstance(
                        error.__cause__,
                        (ExpiredTaskLeaseError, StaleTaskFenceError),
                    ):
                        # The durable start already proves an uncertain external
                        # outcome. A stale holder must not annotate it.
                        return
                    raise
        if fail_known_safe:
            await self._fail_task(operation_id, task_id, guard, error_code)

    async def _fail_task(
        self,
        operation_id: str,
        task_id: str,
        guard: TaskLeaseGuard,
        error_code: str,
    ) -> None:
        async with self._lock:
            state = await self._working_state(operation_id)
            task_index, task = self._task(state, task_id)
            model_call = self._completed_model_call_for_turn(state, task.turn_id)
            if task.status is not TaskStatus.RUNNING:
                raise OperationStateError("task is no longer running")
            placeholder_time = task.updated_at
            state.tasks[task_index] = replace(
                task,
                status=TaskStatus.FAILED,
                updated_at=placeholder_time,
                error_code=error_code,
            )
            lease_index, active_lease = self._lease(state, guard)
            release_placeholder = max(
                timestamp
                for timestamp in (
                    active_lease.acquired_at,
                    active_lease.started_at,
                    active_lease.renewed_at,
                )
                if timestamp is not None
            )
            state.task_leases[lease_index] = replace(
                active_lease,
                released_at=release_placeholder,
                release_reason=error_code,
            )
            self._append_event(
                state,
                "task.failed",
                turn_id=task.turn_id,
                model_call_id=model_call.id,
                call_id=task.call_id,
                task_id=task.id,
                capability_id=task.capability_id,
                executor_id=task.executor_id,
                payload={"task_id": task.id, "error_code": error_code},
            )
            self._append_event(
                state,
                "executor.failed",
                turn_id=task.turn_id,
                model_call_id=model_call.id,
                call_id=task.call_id,
                task_id=task.id,
                capability_id=task.capability_id,
                executor_id=task.executor_id,
                payload={
                    "task_id": task.id,
                    "executor_id": task.executor_id,
                    "error_code": error_code,
                },
            )
            await self._commit_fenced(state, guard)

    def _append_event(
        self,
        state: _OperationState,
        event_type: str,
        *,
        turn_id: str | None = None,
        model_call_id: str | None = None,
        call_id: str | None = None,
        task_id: str | None = None,
        evidence_id: str | None = None,
        approval_id: str | None = None,
        capability_id: str | None = None,
        executor_id: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        state.events.append(
            RuntimeEvent(
                id=self._id_factory("event"),
                type=event_type,
                agent_id=state.operation.agent_id,
                operation_id=state.operation.id,
                session_id=state.operation.session_id,
                turn_id=turn_id,
                model_call_id=model_call_id,
                call_id=call_id,
                task_id=task_id,
                evidence_id=evidence_id,
                approval_id=approval_id,
                capability_id=capability_id,
                executor_id=executor_id,
                payload=payload or {},
                created_at=self._clock(),
            )
        )

    def _fail_locked(
        self,
        state: _OperationState,
        reason: str,
        *,
        final_text: str | None = None,
    ) -> LoopExit:
        now = self._clock()
        state.operation = replace(
            state.operation,
            status=OperationStatus.FAILED,
            updated_at=now,
            final_text=final_text,
            terminal_reason=reason,
        )
        state.loop_state = replace(state.loop_state, phase=LoopPhase.TERMINAL)
        self._append_event(state, "operation.failed", payload={"reason": reason})
        return LoopExit(
            operation_id=state.operation.id,
            kind=LoopExitKind.FAILED,
            reason=reason,
            final_text=final_text,
            created_at=now,
        )

    def _interrupt_locked(
        self,
        state: _OperationState,
        reason: str,
    ) -> LoopExit:
        now = self._clock()
        active_task_index = next(
            (
                index
                for index in range(len(state.tasks) - 1, -1, -1)
                if state.tasks[index].status
                in {
                    TaskStatus.PENDING,
                    TaskStatus.READY,
                    TaskStatus.CLAIMED,
                    TaskStatus.RUNNING,
                    TaskStatus.WAITING_FOR_APPROVAL,
                }
            ),
            None,
        )
        if active_task_index is not None:
            task = state.tasks[active_task_index]
            if task.status is TaskStatus.WAITING_FOR_APPROVAL:
                pending_approval = next(
                    (
                        (index, approval)
                        for index, approval in enumerate(state.approvals)
                        if approval.task_id == task.id
                        and approval.status is ApprovalStatus.PENDING
                    ),
                    None,
                )
                if pending_approval is not None:
                    approval_index, approval = pending_approval
                    state.approvals[approval_index] = replace(
                        approval,
                        status=ApprovalStatus.CANCELLED,
                        decided_at=now,
                        decided_by="runtime:interrupt",
                        decision_reason=reason,
                    )
                    model_call = self._completed_model_call_for_turn(
                        state,
                        task.turn_id,
                    )
                    self._append_event(
                        state,
                        "approval.cancelled",
                        turn_id=task.turn_id,
                        model_call_id=model_call.id,
                        call_id=task.call_id,
                        task_id=task.id,
                        approval_id=approval.id,
                        capability_id=task.capability_id,
                        executor_id=task.executor_id,
                        payload={"approval_id": approval.id, "reason": reason},
                    )
            state.tasks[active_task_index] = replace(
                task,
                status=(
                    TaskStatus.CANCELLED
                    if task.status is TaskStatus.WAITING_FOR_APPROVAL
                    else task.status
                ),
                cancellation_requested=True,
                updated_at=now,
            )
            model_call = self._completed_model_call_for_turn(state, task.turn_id)
            self._append_event(
                state,
                "task.cancellation_requested",
                turn_id=task.turn_id,
                model_call_id=model_call.id,
                call_id=task.call_id,
                task_id=task.id,
                capability_id=task.capability_id,
                executor_id=task.executor_id,
                payload={"reason": reason, "status": task.status.value},
            )
        elif state.model_calls and (
            state.model_calls[-1].status is ModelCallStatus.STARTED
        ):
            model_call = state.model_calls[-1]
            state.model_calls[-1] = replace(
                model_call,
                cancellation_requested=True,
                updated_at=now,
            )
            self._append_event(
                state,
                "model_call.cancellation_requested",
                turn_id=model_call.turn_id,
                model_call_id=model_call.id,
                payload={"model_call_id": model_call.id, "reason": reason},
            )
        state.operation = replace(
            state.operation,
            status=OperationStatus.INTERRUPTED,
            updated_at=now,
            terminal_reason=reason,
        )
        state.loop_state = replace(
            state.loop_state,
            phase=LoopPhase.TERMINAL,
            waiting_approval_id=None,
            interruption_reason=reason,
        )
        self._append_event(
            state,
            "operation.interrupted",
            payload={"reason": reason},
        )
        return LoopExit(
            operation_id=state.operation.id,
            kind=LoopExitKind.INTERRUPTED,
            reason=reason,
            created_at=now,
        )

    @staticmethod
    def _snapshot(state: _OperationState) -> OperationSnapshot:
        return OperationSnapshot(
            trigger=state.trigger,
            operation=state.operation,
            loop_state=state.loop_state,
            budgets=state.budgets,
            turns=tuple(state.turns),
            model_calls=tuple(state.model_calls),
            readiness=tuple(state.readiness),
            tasks=tuple(state.tasks),
            task_dependencies=tuple(state.task_dependencies),
            task_leases=tuple(state.task_leases),
            approvals=tuple(state.approvals),
            evidence=tuple(state.evidence),
            observations=tuple(state.observations),
            events=tuple(state.events),
        )
