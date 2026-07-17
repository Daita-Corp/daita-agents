"""The single provider-neutral generic agent-loop driver."""

from __future__ import annotations

from typing import Protocol

from ..llm.models import ModelRequest, ModelResponse
from ..llm.protocols import ModelProvider
from ..operations.models import AgentTrigger
from ..operations.runtime import OperationRuntime, OperationSnapshot
from .models import LoopExit, LoopPhase, Readiness, Turn


class ContextBuilder(Protocol):
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
    ) -> ModelRequest: ...


class ReadinessEvaluator(Protocol):
    async def evaluate_final_answer(
        self,
        text: str,
        operation: OperationSnapshot,
    ) -> Readiness: ...


class AgentLoop:
    """Own semantic progression while delegating authority to the runtime."""

    def __init__(
        self,
        *,
        runtime: OperationRuntime,
        model: ModelProvider,
        context_builder: ContextBuilder,
        readiness: ReadinessEvaluator,
    ) -> None:
        self._runtime = runtime
        self._model = model
        self._context_builder = context_builder
        self._readiness = readiness

    async def run(self, trigger: AgentTrigger) -> LoopExit:
        operation = await self._runtime.begin(trigger)
        turn = await self._runtime.begin_turn(operation.operation.id)

        try:
            request = await self._context_builder.build(
                await self._runtime.inspect(operation.operation.id),
                turn,
            )
        except Exception:
            return await self._runtime.fail(
                operation.operation.id,
                "context_build_failed",
            )

        model_call = await self._runtime.begin_model_call(
            operation.operation.id,
            turn.id,
            self._model.provider_id,
            request,
        )
        try:
            response = await self._model.generate(request)
        except Exception:
            return await self._runtime.record_model_failure(
                operation.operation.id,
                model_call.id,
                "model_provider_failure",
            )
        if not isinstance(response, ModelResponse):
            return await self._runtime.record_model_failure(
                operation.operation.id,
                model_call.id,
                "malformed_model_response",
            )

        next_phase = (
            LoopPhase.VALIDATING_ACTION
            if response.tool_calls
            else LoopPhase.SYNTHESIZING
        )
        await self._runtime.record_model_response(
            operation.operation.id,
            model_call.id,
            response,
            next_phase=next_phase,
        )

        if response.tool_calls:
            return await self._runtime.fail(
                operation.operation.id,
                "capability_path_not_enabled",
            )

        if response.text is None:
            return await self._runtime.fail(
                operation.operation.id,
                "model_response_missing_text",
            )

        try:
            readiness = await self._readiness.evaluate_final_answer(
                response.text,
                await self._runtime.inspect(operation.operation.id),
            )
        except Exception:
            return await self._runtime.fail(
                operation.operation.id,
                "readiness_evaluation_failed",
            )
        await self._runtime.record_readiness(
            operation.operation.id,
            response.text,
            readiness,
        )
        if not readiness.allowed:
            return await self._runtime.fail(
                operation.operation.id,
                "readiness_not_satisfied",
            )
        return await self._runtime.complete(operation.operation.id, response.text)
