"""The single provider-neutral generic agent-loop driver."""

from __future__ import annotations

from typing import Protocol

from ..llm.models import ModelRequest, ModelResponse, ToolCall, ToolDefinition
from ..llm.protocols import ModelProvider
from ..operations.models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Evidence,
    Observation,
)
from ..operations.runtime import OperationRuntime, OperationSnapshot
from .models import LoopBudgets, LoopExit, LoopPhase, Readiness, Turn


class ContextBuilder(Protocol):
    async def build(
        self,
        operation: OperationSnapshot,
        turn: Turn,
        tools: tuple[ToolDefinition, ...],
    ) -> ModelRequest: ...


class DomainController(Protocol):
    """Supplies domain semantics without owning runtime authority or progression."""

    def tool_views(
        self,
        operation: OperationSnapshot,
    ) -> tuple[ToolDefinition, ...]: ...

    async def validate_action(
        self,
        call: ToolCall,
        operation: OperationSnapshot,
    ) -> ActionProposal | ActionRejection: ...

    async def project_observation(self, evidence: Evidence) -> Observation: ...

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
        domain: DomainController,
        budgets: LoopBudgets = LoopBudgets(),
    ) -> None:
        if not isinstance(budgets, LoopBudgets):
            raise TypeError("budgets must be a LoopBudgets record")
        self._runtime = runtime
        self._model = model
        self._context_builder = context_builder
        self._domain = domain
        self._budgets = budgets

    async def run(self, trigger: AgentTrigger) -> LoopExit:
        operation = await self._runtime.begin(trigger)
        operation_id = operation.operation.id

        while True:
            turn = await self._runtime.begin_turn(operation_id)
            snapshot = await self._runtime.inspect(operation_id)
            try:
                tools = self._domain.tool_views(snapshot)
                request = await self._context_builder.build(snapshot, turn, tools)
                if request.tools != tools:
                    raise ValueError(
                        "context builder changed the domain tool projection"
                    )
                model_call = await self._runtime.begin_model_call(
                    operation_id,
                    turn.id,
                    self._model.provider_id,
                    request,
                )
            except Exception:
                return await self._runtime.fail(
                    operation_id,
                    "context_build_failed",
                )
            try:
                response = await self._model.generate(request)
            except Exception:
                return await self._runtime.record_model_failure(
                    operation_id,
                    model_call.id,
                    "model_provider_failure",
                )
            if not isinstance(response, ModelResponse):
                return await self._runtime.record_model_failure(
                    operation_id,
                    model_call.id,
                    "malformed_model_response",
                )

            next_phase = (
                LoopPhase.VALIDATING_ACTION
                if response.tool_calls
                else LoopPhase.SYNTHESIZING
            )
            await self._runtime.record_model_response(
                operation_id,
                model_call.id,
                response,
                next_phase=next_phase,
            )

            if response.tool_calls:
                terminal = await self._process_actions(
                    operation_id,
                    response.tool_calls,
                )
                if terminal is not None:
                    return terminal
                continue

            if response.text is None:
                return await self._runtime.fail(
                    operation_id,
                    "model_response_missing_text",
                )

            try:
                readiness = await self._domain.evaluate_final_answer(
                    response.text,
                    await self._runtime.inspect(operation_id),
                )
            except Exception:
                return await self._runtime.fail(
                    operation_id,
                    "readiness_evaluation_failed",
                )
            await self._runtime.record_readiness(
                operation_id,
                response.text,
                readiness,
            )
            if not readiness.allowed:
                repaired = await self._runtime.inspect(operation_id)
                if repaired.loop_state.repair_count > self._budgets.max_repairs:
                    return await self._runtime.fail(
                        operation_id,
                        "repair_budget_exhausted",
                    )
                continue
            return await self._runtime.complete(operation_id, response.text)

    async def _process_actions(
        self,
        operation_id: str,
        calls: tuple[ToolCall, ...],
    ) -> LoopExit | None:
        """Process one model response sequentially in declared call order."""

        for call in calls:
            try:
                snapshot = await self._runtime.inspect(operation_id)
                validation = await self._domain.validate_action(
                    call,
                    snapshot,
                )
            except Exception:
                return await self._runtime.fail(
                    operation_id,
                    "action_processing_failed",
                )

            if isinstance(validation, ActionRejection):
                try:
                    await self._runtime.record_action_rejection(
                        operation_id,
                        snapshot.turns[-1].id,
                        call,
                        validation,
                    )
                except Exception:
                    return await self._runtime.fail(
                        operation_id,
                        "action_processing_failed",
                    )
                repaired = await self._runtime.inspect(operation_id)
                if (
                    repaired.loop_state.identical_failure_count
                    >= self._budgets.max_identical_failures
                ):
                    return await self._runtime.fail_no_progress(
                        operation_id,
                        call.id,
                    )
                if repaired.loop_state.repair_count > self._budgets.max_repairs:
                    return await self._runtime.fail(
                        operation_id,
                        "repair_budget_exhausted",
                    )
                return None

            if not isinstance(validation, ActionProposal):
                return await self._runtime.fail(
                    operation_id,
                    "action_processing_failed",
                )

            try:
                evidence = await self._runtime.submit(validation)
                observation = await self._domain.project_observation(evidence)
                await self._runtime.append_observation(observation)
            except Exception:
                return await self._runtime.fail(
                    operation_id,
                    "action_processing_failed",
                )
        return None
