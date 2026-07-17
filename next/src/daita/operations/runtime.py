"""In-memory operation runtime used by the Phase 1 loop laboratory."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
import hashlib
import math
from uuid import uuid4

from .._json import FrozenJsonObject, canonical_json
from ..capabilities import (
    CapabilityExecutionError,
    CapabilityRegistry,
    EvidenceCandidate,
    EvidenceValidationError,
    ExecutionRequest,
)
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
from .models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Evidence,
    Observation,
    Operation,
    OperationStatus,
    Task,
    TaskStatus,
    TriggerKind,
)


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


class ModelCallStatus(str, Enum):
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    id: str
    type: str
    agent_id: str
    operation_id: str
    created_at: datetime
    turn_id: str | None = None
    call_id: str | None = None
    task_id: str | None = None
    evidence_id: str | None = None
    capability_id: str | None = None
    executor_id: str | None = None
    payload: Mapping[str, object] = FrozenJsonObject(())

    def __post_init__(self) -> None:
        for field_name, value in (
            ("turn_id", self.turn_id),
            ("call_id", self.call_id),
            ("task_id", self.task_id),
            ("evidence_id", self.evidence_id),
            ("capability_id", self.capability_id),
            ("executor_id", self.executor_id),
        ):
            if value is not None:
                _required_text(value, f"event {field_name}")
        object.__setattr__(self, "payload", FrozenJsonObject.from_mapping(self.payload))


@dataclass(frozen=True, slots=True)
class ModelCall:
    id: str
    operation_id: str
    turn_id: str
    provider_id: str
    request: ModelRequest
    status: ModelCallStatus
    created_at: datetime
    updated_at: datetime
    response: ModelResponse | None = None
    error_code: str | None = None
    cancellation_requested: bool = False


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    trigger: AgentTrigger
    operation: Operation
    loop_state: LoopState
    budgets: LoopBudgets
    turns: tuple[Turn, ...]
    model_calls: tuple[ModelCall, ...]
    readiness: tuple[Readiness, ...]
    tasks: tuple[Task, ...]
    evidence: tuple[Evidence, ...]
    observations: tuple[Observation, ...]
    events: tuple[RuntimeEvent, ...]


@dataclass(slots=True)
class _OperationState:
    trigger: AgentTrigger
    operation: Operation
    loop_state: LoopState
    budgets: LoopBudgets
    turns: list[Turn]
    model_calls: list[ModelCall]
    readiness: list[Readiness]
    tasks: list[Task]
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


class OperationRuntime:
    """Commit inspectable operation/loop transitions under one in-memory lock."""

    def __init__(
        self,
        *,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _random_id,
        capabilities: CapabilityRegistry | None = None,
    ) -> None:
        self._clock = clock
        self._id_factory = id_factory
        self._capabilities = capabilities or CapabilityRegistry()
        self._lock = asyncio.Lock()
        self._states: dict[str, _OperationState] = {}
        self._operation_by_trigger: dict[str, str] = {}

    async def begin(
        self,
        trigger: AgentTrigger,
        *,
        budgets: LoopBudgets = LoopBudgets(),
    ) -> OperationSnapshot:
        async with self._lock:
            if not isinstance(budgets, LoopBudgets):
                raise TypeError("budgets must be a LoopBudgets record")
            existing_id = self._operation_by_trigger.get(trigger.id)
            if existing_id is not None:
                raise OperationStateError(
                    f"trigger already owns operation: {existing_id}; "
                    "resume is introduced with persistent recovery in Phase 2"
                )
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
                trigger=trigger,
                operation=operation,
                loop_state=LoopState(phase=LoopPhase.PREPARING_CONTEXT),
                budgets=budgets,
                turns=[],
                model_calls=[],
                readiness=[],
                tasks=[],
                evidence=[],
                observations=[],
                events=[],
            )
            self._append_event(state, "trigger.received")
            self._append_event(state, "operation.created")
            self._states[operation_id] = state
            self._operation_by_trigger[trigger.id] = operation_id
            return self._snapshot(state)

    async def begin_turn(self, operation_id: str) -> Turn:
        async with self._lock:
            state = self._working_state(operation_id)
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
            self._commit(state)
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
            state = self._working_state(operation_id)
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
            self._append_event(state, "context.built", turn_id=turn_id)
            self._append_event(
                state,
                "model_call.started",
                turn_id=turn_id,
                payload={"model_call_id": model_call.id, "provider_id": provider_id},
            )
            self._commit(state)
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
            state = self._working_state(operation_id)
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
                payload={
                    "finish_reason": response.finish_reason.value,
                    "model_call_id": model_call.id,
                    "total_tokens": response.usage.total_tokens,
                    "estimated_cost_usd": str(response.usage.estimated_cost_usd),
                },
            )
            self._commit(state)

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
            state = self._working_state(operation_id)
            if not state.model_calls or (
                state.model_calls[-1].status is not ModelCallStatus.COMPLETED
            ):
                raise OperationStateError("readiness requires a completed model call")
            model_response = state.model_calls[-1].response
            if (
                model_response is None
                or model_response.tool_calls
                or model_response.text != final_text
            ):
                raise OperationStateError(
                    "readiness text must match the committed final model response"
                )
            readiness_turn_id = state.model_calls[-1].turn_id
            if any(
                event.type == "readiness.recorded"
                and event.turn_id == readiness_turn_id
                for event in state.events
            ):
                raise OperationStateError(
                    "model response already has a readiness decision"
                )
            if any(
                task.status in {TaskStatus.PENDING, TaskStatus.RUNNING}
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
                    turn_id=state.model_calls[-1].turn_id,
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
                turn_id=state.model_calls[-1].turn_id,
                payload={"allowed": readiness.allowed, "code": readiness.code},
            )
            if correction is not None:
                self._append_event(
                    state,
                    "observation.recorded",
                    turn_id=correction.turn_id,
                    payload={
                        "code": correction.code,
                        "repair": "readiness",
                        "truncated": correction.truncated,
                    },
                )
            self._commit(state)
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
            state = self._working_state(operation_id)
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
                    call_id=skipped_call.id,
                    payload={
                        "code": skipped.code,
                        "repair": "action_skip",
                        "truncated": skipped.truncated,
                    },
                )
            self._commit(state)
            return observation

    async def submit(
        self,
        proposal: ActionProposal,
        *,
        timeout_seconds: float | None = None,
    ) -> Evidence:
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

        # Commit the unexecuted task first. The proposal is untrusted until it
        # is bound to an exact tool call in a committed model response.
        async with self._lock:
            state = self._working_state(proposal.operation_id)
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
            task = Task(
                id=self._id_factory("task"),
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
            self._commit(state)

        # A separate atomic transition makes RUNNING and executor.started
        # durable before the only external execution call in production code.
        wall_deadline_error: OperationWallTimeExceeded | None = None
        execution_timeout_seconds = 0.0
        timeout_is_wall = False
        async with self._lock:
            state = self._working_state(task.operation_id)
            task_index, committed_task = self._task(state, task.id)
            if committed_task.status is not TaskStatus.PENDING:
                raise OperationStateError("task is no longer pending")
            capability, executor = self._capabilities.resolve_execution(
                committed_task.capability_id
            )
            if capability.executor_id != committed_task.executor_id:
                raise OperationStateError("task executor identity changed")
            now = self._clock()
            task = replace(
                committed_task,
                status=TaskStatus.RUNNING,
                updated_at=now,
            )
            state.tasks[task_index] = task
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "executor.started",
                turn_id=task.turn_id,
                call_id=task.call_id,
                task_id=task.id,
                capability_id=task.capability_id,
                executor_id=task.executor_id,
                payload={"task_id": task.id, "executor_id": task.executor_id},
            )
            deadline_now = self._clock()
            if deadline_now.tzinfo is None or deadline_now.utcoffset() is None:
                raise ValueError("runtime clock must return a timezone-aware datetime")
            elapsed_seconds = max(
                0.0,
                (deadline_now - state.operation.created_at).total_seconds(),
            )
            remaining_wall_time = state.budgets.max_wall_time_seconds - elapsed_seconds
            if remaining_wall_time <= 0:
                # Discard the uncommitted RUNNING/executor.started projection.
                # The persisted task proves that execution was blocked before
                # the executor invocation boundary.
                state = self._working_state(task.operation_id)
                task_index, pending_task = self._task(state, task.id)
                state.tasks[task_index] = replace(
                    pending_task,
                    status=TaskStatus.FAILED,
                    updated_at=deadline_now,
                    error_code="task_timeout",
                )
                state.operation = replace(
                    state.operation,
                    updated_at=deadline_now,
                )
                self._append_event(
                    state,
                    "task.failed",
                    turn_id=pending_task.turn_id,
                    call_id=pending_task.call_id,
                    task_id=pending_task.id,
                    capability_id=pending_task.capability_id,
                    executor_id=pending_task.executor_id,
                    payload={
                        "task_id": pending_task.id,
                        "error_code": "task_timeout",
                    },
                )
                self._commit(state)
                wall_deadline_error = OperationWallTimeExceeded(task.id)
            else:
                timeout_candidates = [
                    state.budgets.task_timeout_seconds,
                    remaining_wall_time,
                ]
                if requested_timeout_seconds is not None:
                    timeout_candidates.append(requested_timeout_seconds)
                execution_timeout_seconds = min(timeout_candidates)
                timeout_is_wall = remaining_wall_time <= min(
                    state.budgets.task_timeout_seconds,
                    (
                        requested_timeout_seconds
                        if requested_timeout_seconds is not None
                        else math.inf
                    ),
                )
                self._commit(state)

        if wall_deadline_error is not None:
            raise wall_deadline_error

        request = ExecutionRequest(
            operation_id=task.operation_id,
            task_id=task.id,
            turn_id=task.turn_id,
            capability_id=task.capability_id,
            attempt=task.attempt,
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
                await self._fail_task(task.id, "executor_failed")
                raise CapabilityExecutionError(
                    f"executor failed for task {task.id}"
                ) from error

        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError
        if timeout.expired():
            await self._fail_task(task.id, "task_timeout")
            if timeout_is_wall:
                raise OperationWallTimeExceeded(task.id) from deadline_cause
            raise TaskExecutionTimeout(
                task.id,
                execution_timeout_seconds,
            ) from deadline_cause

        if deadline_cause is not None:
            # ``timeout.expired()`` is stable once the context has exited, so
            # this branch is defensive against an invalid timeout lifecycle.
            await self._fail_task(task.id, "executor_failed")
            raise CapabilityExecutionError(
                f"executor failed for task {task.id}"
            ) from deadline_cause

        execution_identity_error: Exception | None = None
        accepted_candidate: EvidenceCandidate | None = None
        validation_error: EvidenceValidationError | None = None
        async with self._lock:
            state = self._working_state(task.operation_id)
            task_index, committed_task = self._task(state, task.id)
            if committed_task.status is not TaskStatus.RUNNING:
                raise OperationStateError("task is no longer running")
            try:
                current_capability, current_executor = (
                    self._capabilities.resolve_execution(committed_task.capability_id)
                )
            except ValueError as error:
                execution_identity_error = error
            else:
                if (
                    current_capability.executor_id != committed_task.executor_id
                    or current_executor is not executor
                ):
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

            if accepted_candidate is None:
                # Do not commit this working copy. _fail_task publishes a
                # terminal task with no evidence from the prior committed state.
                pass
            else:
                now = self._clock()
                payload_json = canonical_json(accepted_candidate.payload)
                evidence = Evidence(
                    id=self._id_factory("evidence"),
                    operation_id=task.operation_id,
                    task_id=task.id,
                    turn_id=task.turn_id,
                    capability_id=task.capability_id,
                    executor_id=task.executor_id,
                    kind=accepted_candidate.kind,
                    schema_version=accepted_candidate.schema_version,
                    attempt=task.attempt,
                    accepted=True,
                    payload=accepted_candidate.payload,
                    content_hash=(
                        "sha256:"
                        + hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
                    ),
                    created_at=now,
                )
                state.evidence.append(evidence)
                state.tasks[task_index] = replace(
                    committed_task,
                    status=TaskStatus.SUCCEEDED,
                    evidence_ids=(evidence.id,),
                    updated_at=now,
                )
                state.loop_state = replace(
                    state.loop_state,
                    phase=LoopPhase.OBSERVING,
                    identical_failure_count=0,
                    no_progress_fingerprints=(),
                )
                state.operation = replace(state.operation, updated_at=now)
                self._append_event(
                    state,
                    "executor.completed",
                    turn_id=task.turn_id,
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
                    call_id=task.call_id,
                    task_id=task.id,
                    evidence_id=evidence.id,
                    capability_id=task.capability_id,
                    executor_id=task.executor_id,
                    payload={"task_id": task.id},
                )
                self._commit(state)
                return evidence

        if execution_identity_error is not None:
            await self._fail_task(task.id, "execution_identity_changed")
            raise CapabilityExecutionError(
                f"executor identity changed for task {task.id}"
            ) from execution_identity_error
        await self._fail_task(task.id, "evidence_rejected")
        assert validation_error is not None
        raise validation_error

    async def append_observation(self, observation: Observation) -> None:
        if not isinstance(observation, Observation):
            raise TypeError("observation must be an Observation")
        async with self._lock:
            state = self._working_state(observation.operation_id)
            if observation.evidence_id is None or observation.task_id is None:
                raise OperationStateError(
                    "executor observation requires task and evidence identity"
                )
            _, task = self._task(state, observation.task_id)
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
            self._commit(state)

    async def complete(self, operation_id: str, final_text: str) -> LoopExit:
        async with self._lock:
            _required_text(final_text, "final_text")
            state = self._working_state(operation_id)
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
            self._commit(state)
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
            state = self._working_state(operation_id)
            result = self._fail_locked(state, reason, final_text=final_text)
            self._commit(state)
            return result

    async def fail_no_progress(
        self,
        operation_id: str,
        call_id: str,
    ) -> LoopExit:
        _required_text(call_id, "no-progress call_id")
        async with self._lock:
            state = self._working_state(operation_id)
            if not state.turns:
                raise OperationStateError("no-progress failure requires a turn")
            turn_id = state.turns[-1].id
            _, committed_call = self._committed_tool_call(state, turn_id, call_id)
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
                call_id=call_id,
                payload={
                    "count": state.loop_state.identical_failure_count,
                    "fingerprint": fingerprint,
                    "reason": reason,
                },
            )
            result = self._fail_locked(state, reason)
            self._commit(state)
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
            state = self._working_state(operation_id)
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
                    payload={
                        "error_code": reason,
                        "model_call_id": model_call.id,
                    },
                )
            self._append_event(
                state,
                "budget.exhausted",
                turn_id=turn_id,
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
            self._commit(state)
            return result

    async def interrupt(
        self,
        operation_id: str,
        reason: str = "run_cancelled",
    ) -> LoopExit:
        """Persist interruption and active-child cancellation intent atomically."""

        _required_text(reason, "interruption reason")
        async with self._lock:
            state = self._working_state(operation_id)
            now = self._clock()
            active_task_index = next(
                (
                    index
                    for index in range(len(state.tasks) - 1, -1, -1)
                    if state.tasks[index].status
                    in {TaskStatus.PENDING, TaskStatus.RUNNING}
                ),
                None,
            )
            if active_task_index is not None:
                task = state.tasks[active_task_index]
                state.tasks[active_task_index] = replace(
                    task,
                    cancellation_requested=True,
                    updated_at=now,
                )
                self._append_event(
                    state,
                    "task.cancellation_requested",
                    turn_id=task.turn_id,
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
                interruption_reason=reason,
            )
            self._append_event(
                state,
                "operation.interrupted",
                payload={"reason": reason},
            )
            result = LoopExit(
                operation_id=operation_id,
                kind=LoopExitKind.INTERRUPTED,
                reason=reason,
                created_at=now,
            )
            self._commit(state)
            return result

    async def record_model_failure(
        self,
        operation_id: str,
        model_call_id: str,
        error_code: str,
    ) -> LoopExit:
        async with self._lock:
            _required_text(error_code, "model error_code")
            state = self._working_state(operation_id)
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
                payload={"error_code": error_code, "model_call_id": model_call.id},
            )
            result = self._fail_locked(state, error_code)
            self._commit(state)
            return result

    async def inspect(self, operation_id: str) -> OperationSnapshot:
        async with self._lock:
            try:
                state = self._states[operation_id]
            except KeyError as error:
                raise KeyError(f"Unknown operation: {operation_id}") from error
            return self._snapshot(state)

    async def elapsed_seconds(self, operation_id: str) -> float:
        """Return authoritative elapsed wall time from the runtime clock."""

        async with self._lock:
            try:
                state = self._states[operation_id]
            except KeyError as error:
                raise KeyError(f"Unknown operation: {operation_id}") from error
            now = self._clock()
            if now.tzinfo is None or now.utcoffset() is None:
                raise ValueError("runtime clock must return a timezone-aware datetime")
            return max(0.0, (now - state.operation.created_at).total_seconds())

    def _active_state(self, operation_id: str) -> _OperationState:
        try:
            state = self._states[operation_id]
        except KeyError as error:
            raise KeyError(f"Unknown operation: {operation_id}") from error
        if state.operation.status in {
            OperationStatus.SUCCEEDED,
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
            OperationStatus.INTERRUPTED,
        }:
            raise OperationStateError("operation is already terminal")
        return state

    def _working_state(self, operation_id: str) -> _OperationState:
        committed = self._active_state(operation_id)
        return _OperationState(
            trigger=committed.trigger,
            operation=committed.operation,
            loop_state=committed.loop_state,
            budgets=committed.budgets,
            turns=list(committed.turns),
            model_calls=list(committed.model_calls),
            readiness=list(committed.readiness),
            tasks=list(committed.tasks),
            evidence=list(committed.evidence),
            observations=list(committed.observations),
            events=list(committed.events),
        )

    def _commit(self, state: _OperationState) -> None:
        self._states[state.operation.id] = state

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
    def _accepted_evidence(state: _OperationState, evidence_id: str) -> Evidence:
        for evidence in state.evidence:
            if evidence.id == evidence_id and evidence.accepted:
                return evidence
        raise OperationStateError(f"unknown accepted evidence: {evidence_id}")

    async def _fail_task(self, task_id: str, error_code: str) -> None:
        async with self._lock:
            operation_id = self._operation_for_task(task_id)
            state = self._working_state(operation_id)
            task_index, task = self._task(state, task_id)
            if task.status is not TaskStatus.RUNNING:
                raise OperationStateError("task is no longer running")
            now = self._clock()
            state.tasks[task_index] = replace(
                task,
                status=TaskStatus.FAILED,
                updated_at=now,
                error_code=error_code,
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "task.failed",
                turn_id=task.turn_id,
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
            self._commit(state)

    def _operation_for_task(self, task_id: str) -> str:
        for operation_id, state in self._states.items():
            if any(task.id == task_id for task in state.tasks):
                return operation_id
        raise OperationStateError(f"unknown task: {task_id}")

    def _append_event(
        self,
        state: _OperationState,
        event_type: str,
        *,
        turn_id: str | None = None,
        call_id: str | None = None,
        task_id: str | None = None,
        evidence_id: str | None = None,
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
                turn_id=turn_id,
                call_id=call_id,
                task_id=task_id,
                evidence_id=evidence_id,
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
            evidence=tuple(state.evidence),
            observations=tuple(state.observations),
            events=tuple(state.events),
        )
