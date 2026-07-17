"""The single provider-neutral generic agent-loop driver."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

from ..llm.models import ModelRequest, ModelResponse, ToolCall, ToolDefinition
from ..llm.protocols import ModelProvider
from ..operations.models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Evidence,
    Observation,
)
from ..operations.runtime import (
    OperationWallTimeExceeded,
    OperationRuntime,
    OperationSnapshot,
    OperationStateError,
    TaskExecutionTimeout,
)
from .models import LoopBudgets, LoopExit, LoopPhase, Readiness, Turn

_T = TypeVar("_T")


class _WallTimeExhausted(RuntimeError):
    """Internal progression signal translated into a durable budget exit."""


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
        operation = await self._runtime.begin(trigger, budgets=self._budgets)
        operation_id = operation.operation.id

        try:
            try:
                return await self._run_operation(operation_id)
            except _WallTimeExhausted:
                return await self._fail_wall_time(operation_id)
        except asyncio.CancelledError:
            await self._persist_interruption(operation_id)
            raise

    async def _run_operation(self, operation_id: str) -> LoopExit:
        """Run one already-committed operation under its bound budgets."""

        while True:
            terminal = await self._pre_turn_budget_exit(operation_id)
            if terminal is not None:
                return terminal
            turn = await self._runtime.begin_turn(operation_id)
            snapshot = await self._runtime.inspect(operation_id)
            try:
                tools = self._domain.tool_views(snapshot)
                request = await self._await_with_wall_time(
                    operation_id,
                    lambda: self._context_builder.build(snapshot, turn, tools),
                )
                await self._raise_if_wall_exhausted(operation_id)
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
            except _WallTimeExhausted:
                raise
            except Exception:
                return await self._runtime.fail(
                    operation_id,
                    "context_build_failed",
                )
            try:
                response = await self._await_with_wall_time(
                    operation_id,
                    lambda: self._model.generate(request),
                )
            except _WallTimeExhausted:
                raise
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
            terminal = await self._post_response_budget_exit(operation_id)
            if terminal is not None:
                return terminal

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
            final_text = response.text

            try:
                readiness_snapshot = await self._runtime.inspect(operation_id)
                readiness = await self._await_with_wall_time(
                    operation_id,
                    lambda: self._domain.evaluate_final_answer(
                        final_text,
                        readiness_snapshot,
                    ),
                )
            except _WallTimeExhausted:
                raise
            except Exception:
                return await self._runtime.fail(
                    operation_id,
                    "readiness_evaluation_failed",
                )
            correction = await self._runtime.record_readiness(
                operation_id,
                final_text,
                readiness,
            )
            terminal = await self._wall_budget_exit(operation_id)
            if terminal is not None:
                return terminal
            if not readiness.allowed:
                repaired = await self._runtime.inspect(operation_id)
                if repaired.loop_state.repair_count > repaired.budgets.max_repairs:
                    return await self._runtime.fail_budget(
                        operation_id,
                        "repair_budget_exhausted",
                        budget="repairs",
                        limit=repaired.budgets.max_repairs,
                        used=repaired.loop_state.repair_count,
                        turn_id=repaired.turns[-1].id,
                    )
                if (
                    correction is not None
                    and repaired.loop_state.observation_characters
                    > repaired.budgets.max_observation_characters
                ):
                    return await self._runtime.fail_budget(
                        operation_id,
                        "observation_budget_exhausted",
                        budget="observation_characters",
                        limit=repaired.budgets.max_observation_characters,
                        used=repaired.loop_state.observation_characters,
                        turn_id=correction.turn_id,
                    )
                continue
            await self._raise_if_wall_exhausted(operation_id)
            return await self._runtime.complete(operation_id, final_text)

    async def _process_actions(
        self,
        operation_id: str,
        calls: tuple[ToolCall, ...],
    ) -> LoopExit | None:
        """Process one model response sequentially in declared call order."""

        for call in calls:
            terminal = await self._wall_budget_exit(operation_id, call_id=call.id)
            if terminal is not None:
                return terminal
            try:
                snapshot = await self._runtime.inspect(operation_id)
                if snapshot.loop_state.action_count >= snapshot.budgets.max_actions:
                    return await self._runtime.fail_budget(
                        operation_id,
                        "action_budget_exhausted",
                        budget="actions",
                        limit=snapshot.budgets.max_actions,
                        used=snapshot.loop_state.action_count,
                        turn_id=snapshot.turns[-1].id,
                        call_id=call.id,
                    )
                validation = await self._await_with_wall_time(
                    operation_id,
                    lambda: self._domain.validate_action(call, snapshot),
                )
                await self._raise_if_wall_exhausted(operation_id)
            except _WallTimeExhausted:
                raise
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
                terminal = await self._wall_budget_exit(
                    operation_id,
                    call_id=call.id,
                )
                if terminal is not None:
                    return terminal
                repaired = await self._runtime.inspect(operation_id)
                if (
                    repaired.loop_state.identical_failure_count
                    >= repaired.budgets.max_identical_failures
                ):
                    return await self._runtime.fail_no_progress(
                        operation_id,
                        call.id,
                    )
                if repaired.loop_state.repair_count > repaired.budgets.max_repairs:
                    return await self._runtime.fail_budget(
                        operation_id,
                        "repair_budget_exhausted",
                        budget="repairs",
                        limit=repaired.budgets.max_repairs,
                        used=repaired.loop_state.repair_count,
                        turn_id=repaired.turns[-1].id,
                        call_id=call.id,
                    )
                if (
                    repaired.loop_state.observation_characters
                    > repaired.budgets.max_observation_characters
                ):
                    return await self._runtime.fail_budget(
                        operation_id,
                        "observation_budget_exhausted",
                        budget="observation_characters",
                        limit=repaired.budgets.max_observation_characters,
                        used=repaired.loop_state.observation_characters,
                        turn_id=repaired.turns[-1].id,
                        call_id=call.id,
                    )
                return None

            if not isinstance(validation, ActionProposal):
                return await self._runtime.fail(
                    operation_id,
                    "action_processing_failed",
                )

            remaining_wall_time = await self._remaining_wall_time(operation_id)
            try:
                evidence = await self._runtime.submit(
                    validation,
                    timeout_seconds=remaining_wall_time,
                )
            except OperationWallTimeExceeded as error:
                return await self._fail_wall_time(
                    operation_id,
                    call_id=call.id,
                    task_id=error.task_id,
                )
            except TaskExecutionTimeout as error:
                return await self._runtime.fail_budget(
                    operation_id,
                    "task_timeout",
                    budget="task_timeout_seconds",
                    limit=snapshot.budgets.task_timeout_seconds,
                    used=error.timeout_seconds,
                    turn_id=snapshot.turns[-1].id,
                    call_id=call.id,
                    task_id=error.task_id,
                )
            except Exception:
                return await self._runtime.fail(
                    operation_id,
                    "action_processing_failed",
                )

            terminal = await self._wall_budget_exit(
                operation_id,
                call_id=call.id,
                task_id=evidence.task_id,
            )
            if terminal is not None:
                return terminal
            try:
                observation = await self._await_with_wall_time(
                    operation_id,
                    lambda: self._domain.project_observation(evidence),
                )
                await self._raise_if_wall_exhausted(operation_id)
                await self._runtime.append_observation(observation)
            except _WallTimeExhausted:
                raise
            except Exception:
                return await self._runtime.fail(
                    operation_id,
                    "action_processing_failed",
                )
            terminal = await self._wall_budget_exit(
                operation_id,
                call_id=call.id,
                task_id=evidence.task_id,
            )
            if terminal is not None:
                return terminal
            observed = await self._runtime.inspect(operation_id)
            if (
                observed.loop_state.observation_characters
                > observed.budgets.max_observation_characters
            ):
                return await self._runtime.fail_budget(
                    operation_id,
                    "observation_budget_exhausted",
                    budget="observation_characters",
                    limit=observed.budgets.max_observation_characters,
                    used=observed.loop_state.observation_characters,
                    turn_id=observation.turn_id,
                    call_id=call.id,
                    task_id=evidence.task_id,
                )
        return None

    async def _pre_turn_budget_exit(
        self,
        operation_id: str,
    ) -> LoopExit | None:
        terminal = await self._wall_budget_exit(operation_id)
        if terminal is not None:
            return terminal
        snapshot = await self._runtime.inspect(operation_id)
        if snapshot.loop_state.turn_count >= snapshot.budgets.max_turns:
            return await self._runtime.fail_budget(
                operation_id,
                "turn_budget_exhausted",
                budget="turns",
                limit=snapshot.budgets.max_turns,
                used=snapshot.loop_state.turn_count,
            )
        total_tokens = (
            snapshot.loop_state.input_tokens + snapshot.loop_state.output_tokens
        )
        if total_tokens >= snapshot.budgets.max_total_tokens:
            return await self._runtime.fail_budget(
                operation_id,
                "token_budget_exhausted",
                budget="total_tokens",
                limit=snapshot.budgets.max_total_tokens,
                used=total_tokens,
            )
        cost_limit = snapshot.budgets.max_estimated_cost_usd
        if (
            cost_limit is not None
            and snapshot.loop_state.estimated_cost_usd >= cost_limit
        ):
            return await self._runtime.fail_budget(
                operation_id,
                "estimated_cost_budget_exhausted",
                budget="estimated_cost_usd",
                limit=str(cost_limit),
                used=str(snapshot.loop_state.estimated_cost_usd),
            )
        return None

    async def _post_response_budget_exit(
        self,
        operation_id: str,
    ) -> LoopExit | None:
        terminal = await self._wall_budget_exit(operation_id)
        if terminal is not None:
            return terminal
        snapshot = await self._runtime.inspect(operation_id)
        total_tokens = (
            snapshot.loop_state.input_tokens + snapshot.loop_state.output_tokens
        )
        if total_tokens > snapshot.budgets.max_total_tokens:
            return await self._runtime.fail_budget(
                operation_id,
                "token_budget_exhausted",
                budget="total_tokens",
                limit=snapshot.budgets.max_total_tokens,
                used=total_tokens,
                turn_id=snapshot.turns[-1].id,
            )
        cost_limit = snapshot.budgets.max_estimated_cost_usd
        if (
            cost_limit is not None
            and snapshot.loop_state.estimated_cost_usd > cost_limit
        ):
            return await self._runtime.fail_budget(
                operation_id,
                "estimated_cost_budget_exhausted",
                budget="estimated_cost_usd",
                limit=str(cost_limit),
                used=str(snapshot.loop_state.estimated_cost_usd),
                turn_id=snapshot.turns[-1].id,
            )
        return None

    async def _await_with_wall_time(
        self,
        operation_id: str,
        factory: Callable[[], Awaitable[_T]],
    ) -> _T:
        remaining = await self._remaining_wall_time(operation_id)
        timeout = asyncio.timeout(remaining)
        try:
            async with timeout:
                result = await factory()
        except TimeoutError as error:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise asyncio.CancelledError from error
            if timeout.expired():
                raise _WallTimeExhausted from error
            raise
        except Exception as error:
            current_task = asyncio.current_task()
            if current_task is not None and current_task.cancelling():
                raise asyncio.CancelledError from error
            if timeout.expired():
                raise _WallTimeExhausted from error
            raise
        current_task = asyncio.current_task()
        if current_task is not None and current_task.cancelling():
            raise asyncio.CancelledError
        if timeout.expired():
            raise _WallTimeExhausted
        return result

    async def _remaining_wall_time(self, operation_id: str) -> float:
        snapshot = await self._runtime.inspect(operation_id)
        elapsed = await self._runtime.elapsed_seconds(operation_id)
        remaining = snapshot.budgets.max_wall_time_seconds - elapsed
        if remaining <= 0:
            raise _WallTimeExhausted
        return remaining

    async def _raise_if_wall_exhausted(self, operation_id: str) -> None:
        snapshot = await self._runtime.inspect(operation_id)
        elapsed = await self._runtime.elapsed_seconds(operation_id)
        if elapsed >= snapshot.budgets.max_wall_time_seconds:
            raise _WallTimeExhausted

    async def _wall_budget_exit(
        self,
        operation_id: str,
        *,
        call_id: str | None = None,
        task_id: str | None = None,
    ) -> LoopExit | None:
        snapshot = await self._runtime.inspect(operation_id)
        elapsed = await self._runtime.elapsed_seconds(operation_id)
        if elapsed < snapshot.budgets.max_wall_time_seconds:
            return None
        return await self._fail_wall_time(
            operation_id,
            call_id=call_id,
            task_id=task_id,
        )

    async def _fail_wall_time(
        self,
        operation_id: str,
        *,
        call_id: str | None = None,
        task_id: str | None = None,
    ) -> LoopExit:
        snapshot = await self._runtime.inspect(operation_id)
        elapsed = await self._runtime.elapsed_seconds(operation_id)
        limit = snapshot.budgets.max_wall_time_seconds
        return await self._runtime.fail_budget(
            operation_id,
            "wall_time_budget_exhausted",
            budget="wall_time_seconds",
            limit=limit,
            used=max(elapsed, limit),
            turn_id=(snapshot.turns[-1].id if snapshot.turns else None),
            call_id=call_id,
            task_id=task_id,
        )

    async def _persist_interruption(self, operation_id: str) -> None:
        commit = asyncio.create_task(self._runtime.interrupt(operation_id))
        while not commit.done():
            try:
                await asyncio.shield(commit)
            except asyncio.CancelledError:
                continue
        try:
            commit.result()
        except OperationStateError:
            # A terminal commit racing cancellation remains authoritative.
            return
