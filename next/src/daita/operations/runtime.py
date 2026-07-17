"""In-memory operation runtime used by the Phase 1 loop laboratory."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from uuid import uuid4

from .._json import FrozenJsonObject
from ..llm.models import ModelRequest, ModelResponse
from ..loop.models import (
    LoopExit,
    LoopExitKind,
    LoopPhase,
    LoopState,
    Readiness,
    Turn,
)
from .models import (
    AgentTrigger,
    Observation,
    Operation,
    OperationStatus,
    TriggerKind,
)


class OperationStateError(RuntimeError):
    """Raised when a runtime transition conflicts with committed state."""


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
    payload: Mapping[str, object] = FrozenJsonObject(())

    def __post_init__(self) -> None:
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


@dataclass(frozen=True, slots=True)
class OperationSnapshot:
    trigger: AgentTrigger
    operation: Operation
    loop_state: LoopState
    turns: tuple[Turn, ...]
    model_calls: tuple[ModelCall, ...]
    readiness: tuple[Readiness, ...]
    observations: tuple[Observation, ...]
    events: tuple[RuntimeEvent, ...]


@dataclass(slots=True)
class _OperationState:
    trigger: AgentTrigger
    operation: Operation
    loop_state: LoopState
    turns: list[Turn]
    model_calls: list[ModelCall]
    readiness: list[Readiness]
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
    ) -> None:
        self._clock = clock
        self._id_factory = id_factory
        self._lock = asyncio.Lock()
        self._states: dict[str, _OperationState] = {}
        self._operation_by_trigger: dict[str, str] = {}

    async def begin(self, trigger: AgentTrigger) -> OperationSnapshot:
        async with self._lock:
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
                turns=[],
                model_calls=[],
                readiness=[],
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
    ) -> None:
        async with self._lock:
            _required_text(final_text, "final_text")
            if not isinstance(readiness, Readiness):
                raise TypeError("readiness must be a Readiness record")
            state = self._working_state(operation_id)
            if not state.model_calls or (
                state.model_calls[-1].status is not ModelCallStatus.COMPLETED
            ):
                raise OperationStateError("readiness requires a completed model call")
            now = self._clock()
            state.readiness.append(readiness)
            state.loop_state = replace(
                state.loop_state,
                phase=LoopPhase.SYNTHESIZING,
                final_answer_candidate=final_text,
            )
            state.operation = replace(state.operation, updated_at=now)
            self._append_event(
                state,
                "readiness.recorded",
                turn_id=state.model_calls[-1].turn_id,
                payload={"allowed": readiness.allowed, "code": readiness.code},
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
            turns=list(committed.turns),
            model_calls=list(committed.model_calls),
            readiness=list(committed.readiness),
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

    def _append_event(
        self,
        state: _OperationState,
        event_type: str,
        *,
        turn_id: str | None = None,
        payload: Mapping[str, object] | None = None,
    ) -> None:
        state.events.append(
            RuntimeEvent(
                id=self._id_factory("event"),
                type=event_type,
                agent_id=state.operation.agent_id,
                operation_id=state.operation.id,
                turn_id=turn_id,
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
            turns=tuple(state.turns),
            model_calls=tuple(state.model_calls),
            readiness=tuple(state.readiness),
            observations=tuple(state.observations),
            events=tuple(state.events),
        )
