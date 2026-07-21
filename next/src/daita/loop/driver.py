"""The single provider-neutral generic agent-loop driver."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Protocol, TypeVar

from ..context.budgeting import RequiredContextOverflow
from ..llm.errors import ModelProviderError, ProviderErrorCode
from ..llm.models import ModelRequest, ModelResponse, ToolCall, ToolDefinition
from ..llm.protocols import ModelProvider
from ..operations.checkpoints import ModelCall, ModelCallStatus, OperationSnapshot
from ..operations.models import (
    ActionProposal,
    ActionRejection,
    AgentTrigger,
    Evidence,
    Observation,
    OperationStatus,
    TaskStatus,
)
from ..operations.runtime import (
    OperationWallTimeExceeded,
    OperationRuntime,
    OperationStateError,
    TaskExecutionTimeout,
    TaskOutcomeUnknown,
)
from .models import LoopBudgets, LoopExit, LoopExitKind, LoopPhase, Readiness, Turn

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

    async def tool_views(
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

    async def run(
        self,
        trigger: AgentTrigger,
        *,
        budgets: LoopBudgets | None = None,
    ) -> LoopExit:
        effective_budgets = self._budgets if budgets is None else budgets
        if not isinstance(effective_budgets, LoopBudgets):
            raise TypeError("budgets must be a LoopBudgets record or None")
        operation = await self._runtime.begin(trigger, budgets=effective_budgets)
        return await self._continue_operation(operation.operation.id)

    async def resume(self, operation_id: str) -> LoopExit:
        """Continue one known operation from its authoritative checkpoint."""

        return await self._continue_operation(operation_id)

    async def recover_startup(self, agent_id: str) -> tuple[LoopExit, ...]:
        """Resume one ordered snapshot of an agent's nonterminal operations."""

        snapshots = await self._runtime.inspect_nonterminal(agent_id)
        results: list[LoopExit] = []
        for snapshot in snapshots:
            results.append(await self.resume(snapshot.operation.id))
        return tuple(results)

    async def _continue_operation(self, operation_id: str) -> LoopExit:
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
            snapshot = await self._runtime.inspect(operation_id)
            if snapshot.operation.status is OperationStatus.WAITING_FOR_APPROVAL:
                if await self._runtime.resume_approval(operation_id):
                    continue
            checkpoint_exit = self._checkpoint_exit(snapshot)
            if checkpoint_exit is not None:
                return checkpoint_exit

            if not snapshot.turns:
                terminal = await self._begin_next_turn(operation_id)
                if terminal is not None:
                    return terminal
                continue

            turn = snapshot.turns[-1]
            if turn.model_request_id is None:
                terminal = await self._begin_model_call(snapshot, turn)
                if terminal is not None:
                    return terminal
                continue

            model_call = next(
                call
                for call in snapshot.model_calls
                if call.id == turn.model_request_id
            )
            if model_call.status is ModelCallStatus.STARTED:
                terminal = await self._continue_model_call(model_call)
                if terminal is not None:
                    return terminal
                continue
            if model_call.status is ModelCallStatus.FAILED:
                return await self._runtime.fail(
                    operation_id,
                    model_call.error_code or "model_provider_failure",
                )

            terminal = await self._process_response(operation_id, model_call)
            if terminal is not None:
                return terminal
            terminal = await self._begin_next_turn(operation_id)
            if terminal is not None:
                return terminal

    async def _begin_next_turn(self, operation_id: str) -> LoopExit | None:
        terminal = await self._pre_turn_budget_exit(operation_id)
        if terminal is not None:
            return terminal
        await self._runtime.begin_turn(operation_id)
        return None

    async def _begin_model_call(
        self,
        snapshot: OperationSnapshot,
        turn: Turn,
    ) -> LoopExit | None:
        """Build context only for a committed turn that has no request."""

        terminal = await self._wall_budget_exit(snapshot.operation.id)
        if terminal is not None:
            return terminal
        try:
            tools = await self._await_with_wall_time(
                snapshot.operation.id,
                lambda: self._domain.tool_views(snapshot),
            )
            request = await self._await_with_wall_time(
                snapshot.operation.id,
                lambda: self._context_builder.build(snapshot, turn, tools),
            )
            await self._raise_if_wall_exhausted(snapshot.operation.id)
            if request.tools != tools:
                raise ValueError("context builder changed the domain tool projection")
            await self._runtime.begin_model_call(
                snapshot.operation.id,
                turn.id,
                self._model.provider_id,
                request,
            )
        except _WallTimeExhausted:
            raise
        except RequiredContextOverflow as error:
            return await self._runtime.fail_required_context(
                snapshot.operation.id,
                profile_id=error.profile_id,
                input_limit_tokens=error.input_limit_tokens,
                output_reserve_tokens=error.output_reserve_tokens,
                tool_tokens=error.tool_tokens,
                required_system_tokens=error.required_system_tokens,
                required_routing_tokens=error.required_routing_tokens,
                required_intent_tokens=error.required_intent_tokens,
                current_operation_envelope_tokens=(
                    error.current_operation_envelope_tokens
                ),
                current_operation_body_tokens=error.current_operation_body_tokens,
                minimum_session_tokens=error.minimum_session_tokens,
                projected_session_tokens=error.projected_session_tokens,
                required_tokens=error.required_tokens,
                available_tokens=error.available_tokens,
                total_required_tokens=error.total_required_tokens,
                optional_omitted_tokens=error.optional_omitted_tokens,
            )
        except Exception:
            return await self._runtime.fail(
                snapshot.operation.id,
                "context_build_failed",
            )
        return None

    async def _continue_model_call(self, model_call: ModelCall) -> LoopExit | None:
        """Resend an exact STARTED request under at-least-once inference."""

        operation_id = model_call.operation_id
        if model_call.provider_id != self._model.provider_id:
            return await self._runtime.record_model_failure(
                operation_id,
                model_call.id,
                "model_provider_identity_changed",
            )
        if not _provider_supports_request_policy(self._model, model_call.request):
            return await self._runtime.record_model_failure(
                operation_id,
                model_call.id,
                ProviderErrorCode.INVALID_REQUEST.value,
            )
        try:
            response = await self._await_with_wall_time(
                operation_id,
                lambda: self._model.generate(model_call.request),
            )
        except _WallTimeExhausted:
            raise
        except ModelProviderError as error:
            return await self._runtime.record_model_failure(
                operation_id,
                model_call.id,
                error.code.value,
                routing=error.routing,
            )
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
        return None

    @staticmethod
    def _checkpoint_exit(snapshot: OperationSnapshot) -> LoopExit | None:
        """Project persisted terminal/waiting operation state without mutation."""

        kind = {
            OperationStatus.SUCCEEDED: LoopExitKind.COMPLETED,
            OperationStatus.FAILED: LoopExitKind.FAILED,
            OperationStatus.CANCELLED: LoopExitKind.CANCELLED,
            OperationStatus.INTERRUPTED: LoopExitKind.INTERRUPTED,
        }.get(snapshot.operation.status)
        if kind is not None:
            return LoopExit(
                operation_id=snapshot.operation.id,
                kind=kind,
                reason=snapshot.operation.terminal_reason or kind.value,
                final_text=snapshot.operation.final_text,
                created_at=snapshot.operation.updated_at,
            )
        waiting_reason = {
            OperationStatus.WAITING_FOR_APPROVAL: "operation_waiting_for_approval",
            OperationStatus.WAITING_FOR_INPUT: "operation_waiting_for_input",
        }.get(snapshot.operation.status)
        if waiting_reason is None:
            return None
        return LoopExit(
            operation_id=snapshot.operation.id,
            kind=LoopExitKind.WAITING,
            reason=waiting_reason,
            created_at=snapshot.operation.updated_at,
        )

    async def _process_response(
        self,
        operation_id: str,
        model_call: ModelCall,
    ) -> LoopExit | None:
        """Advance one already-committed normalized model response."""

        response = model_call.response
        if response is None:
            return await self._runtime.fail(
                operation_id,
                "checkpoint_recovery_failed",
            )

        if response.tool_calls:
            return await self._process_actions(
                operation_id,
                model_call,
            )

        terminal = await self._post_response_budget_exit(operation_id)
        if terminal is not None:
            return terminal

        if response.text is None:
            return await self._runtime.fail(
                operation_id,
                "model_response_missing_text",
            )
        final_text = response.text

        readiness_snapshot = await self._runtime.inspect(operation_id)
        readiness_event = next(
            (
                event
                for event in readiness_snapshot.events
                if event.type == "readiness.recorded"
                and event.model_call_id == model_call.id
            ),
            None,
        )
        if readiness_event is not None:
            if not readiness_snapshot.readiness:
                return await self._runtime.fail(
                    operation_id,
                    "checkpoint_recovery_failed",
                )
            readiness = readiness_snapshot.readiness[-1]
            if (
                readiness_event.payload.get("allowed") is not readiness.allowed
                or readiness_event.payload.get("code") != readiness.code
            ):
                return await self._runtime.fail(
                    operation_id,
                    "checkpoint_recovery_failed",
                )
        else:
            try:
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
            await self._runtime.record_readiness(
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
                    turn_id=model_call.turn_id,
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
                    turn_id=model_call.turn_id,
                )
            return None
        await self._raise_if_wall_exhausted(operation_id)
        return await self._runtime.complete(operation_id, final_text)

    async def _process_actions(
        self,
        operation_id: str,
        model_call: ModelCall,
    ) -> LoopExit | None:
        """Process one model response sequentially in declared call order."""

        response = model_call.response
        if response is None:
            return await self._runtime.fail(
                operation_id,
                "checkpoint_recovery_failed",
            )

        for call_index, call in enumerate(response.tool_calls):
            snapshot = await self._runtime.inspect(operation_id)
            existing_task = next(
                (
                    task
                    for task in snapshot.tasks
                    if task.turn_id == model_call.turn_id and task.call_id == call.id
                ),
                None,
            )
            if existing_task is not None:
                failed_observation = next(
                    (
                        observation
                        for observation in snapshot.observations
                        if observation.task_id == existing_task.id
                        and not observation.success
                    ),
                    None,
                )
                if failed_observation is not None and existing_task.status in {
                    TaskStatus.FAILED,
                    TaskStatus.CANCELLED,
                }:
                    return None
                observed_evidence_ids = {
                    observation.evidence_id
                    for observation in snapshot.observations
                    if observation.task_id == existing_task.id
                    and observation.evidence_id is not None
                }
                if existing_task.evidence_ids and all(
                    evidence_id in observed_evidence_ids
                    for evidence_id in existing_task.evidence_ids
                ):
                    later_call_ids = {
                        later_call.id
                        for later_call in response.tool_calls[call_index + 1 :]
                    }
                    task_call_ids = {task.id: task.call_id for task in snapshot.tasks}
                    has_later_progress = any(
                        task.turn_id == model_call.turn_id
                        and task.call_id in later_call_ids
                        for task in snapshot.tasks
                    ) or any(
                        observation.turn_id == model_call.turn_id
                        and (
                            observation.call_id in later_call_ids
                            or task_call_ids.get(observation.task_id or "")
                            in later_call_ids
                        )
                        for observation in snapshot.observations
                    )
                    if not has_later_progress:
                        terminal = await self._after_task_observation(
                            operation_id,
                            model_call.turn_id,
                            call.id,
                            existing_task.id,
                        )
                        if terminal is not None:
                            return terminal
                    continue
            else:
                rejection_event = next(
                    (
                        event
                        for event in snapshot.events
                        if event.type == "action.rejected"
                        and event.model_call_id == model_call.id
                        and event.call_id == call.id
                    ),
                    None,
                )
                if rejection_event is not None:
                    return await self._after_action_rejection(
                        operation_id,
                        model_call.turn_id,
                        call.id,
                    )
                if any(
                    observation.turn_id == model_call.turn_id
                    and observation.call_id == call.id
                    for observation in snapshot.observations
                ):
                    continue

            if existing_task is None:
                terminal = await self._post_response_budget_exit(operation_id)
                if terminal is not None:
                    return terminal
                if snapshot.loop_state.action_count >= snapshot.budgets.max_actions:
                    return await self._runtime.fail_budget(
                        operation_id,
                        "action_budget_exhausted",
                        budget="actions",
                        limit=snapshot.budgets.max_actions,
                        used=snapshot.loop_state.action_count,
                        turn_id=model_call.turn_id,
                        call_id=call.id,
                    )
                try:
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

                # The provider may emit a registered tool that was not exposed in
                # this exact request.  Domain validation still runs so projection
                # remains presentation metadata rather than an authorization
                # grant, but an unexposed call must become model-visible repair
                # evidence before the runtime can materialize a task.
                if isinstance(validation, ActionProposal) and call.name not in {
                    definition.name for definition in model_call.request.tools
                }:
                    validation = ActionRejection(
                        code="action.tool_not_projected",
                        message=(
                            "The requested tool was not available for this model "
                            "request."
                        ),
                        details={
                            "capability_id": validation.capability_id,
                            "tool_name": call.name,
                        },
                    )

                if isinstance(validation, ActionRejection):
                    try:
                        await self._runtime.record_action_rejection(
                            operation_id,
                            model_call.turn_id,
                            call,
                            validation,
                        )
                    except Exception:
                        return await self._runtime.fail(
                            operation_id,
                            "action_processing_failed",
                        )
                    return await self._after_action_rejection(
                        operation_id,
                        model_call.turn_id,
                        call.id,
                    )

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
                except TaskOutcomeUnknown as error:
                    current = await self._runtime.inspect(operation_id)
                    return await self._task_checkpoint_exit(
                        current,
                        error.task_id,
                        outcome_unknown=True,
                    )
                except OperationWallTimeExceeded as error:
                    current = await self._runtime.inspect(operation_id)
                    task = next(
                        task for task in current.tasks if task.id == error.task_id
                    )
                    if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                        return await self._task_checkpoint_exit(current, task.id)
                    return await self._fail_wall_time(
                        operation_id,
                        call_id=call.id,
                        task_id=error.task_id,
                    )
                except TaskExecutionTimeout as error:
                    current = await self._runtime.inspect(operation_id)
                    task = next(
                        task for task in current.tasks if task.id == error.task_id
                    )
                    if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                        return await self._task_checkpoint_exit(current, task.id)
                    return await self._runtime.fail_budget(
                        operation_id,
                        "task_timeout",
                        budget="task_timeout_seconds",
                        limit=snapshot.budgets.task_timeout_seconds,
                        used=error.timeout_seconds,
                        turn_id=model_call.turn_id,
                        call_id=call.id,
                        task_id=error.task_id,
                    )
                except Exception:
                    current = await self._runtime.inspect(operation_id)
                    checkpoint_task = next(
                        (
                            task
                            for task in current.tasks
                            if task.turn_id == model_call.turn_id
                            and task.call_id == call.id
                        ),
                        None,
                    )
                    if checkpoint_task is not None and checkpoint_task.status in {
                        TaskStatus.CLAIMED,
                        TaskStatus.RUNNING,
                        TaskStatus.MANUAL_RECOVERY_REQUIRED,
                    }:
                        return await self._task_checkpoint_exit(
                            current,
                            checkpoint_task.id,
                        )
                    return await self._runtime.fail(
                        operation_id,
                        "action_processing_failed",
                    )
                if evidence is None:
                    current = await self._runtime.inspect(operation_id)
                    task = next(
                        task
                        for task in current.tasks
                        if task.turn_id == model_call.turn_id
                        and task.call_id == call.id
                    )
                    if task.status in {TaskStatus.FAILED, TaskStatus.CANCELLED} and any(
                        observation.task_id == task.id and not observation.success
                        for observation in current.observations
                    ):
                        return None
                    return await self._task_checkpoint_exit(current, task.id)
            else:
                try:
                    resumed_evidence = await self._runtime.resume_task(
                        operation_id,
                        existing_task.id,
                    )
                except TaskOutcomeUnknown as error:
                    current = await self._runtime.inspect(operation_id)
                    return await self._task_checkpoint_exit(
                        current,
                        error.task_id,
                        outcome_unknown=True,
                    )
                except OperationWallTimeExceeded as error:
                    current = await self._runtime.inspect(operation_id)
                    task = next(
                        task for task in current.tasks if task.id == error.task_id
                    )
                    if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                        return await self._task_checkpoint_exit(current, task.id)
                    return await self._fail_wall_time(
                        operation_id,
                        call_id=call.id,
                        task_id=error.task_id,
                    )
                except TaskExecutionTimeout as error:
                    current = await self._runtime.inspect(operation_id)
                    task = next(
                        task for task in current.tasks if task.id == error.task_id
                    )
                    if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
                        return await self._task_checkpoint_exit(current, task.id)
                    return await self._runtime.fail_budget(
                        operation_id,
                        "task_timeout",
                        budget="task_timeout_seconds",
                        limit=snapshot.budgets.task_timeout_seconds,
                        used=error.timeout_seconds,
                        turn_id=model_call.turn_id,
                        call_id=call.id,
                        task_id=error.task_id,
                    )
                except OperationStateError:
                    current = await self._runtime.inspect(operation_id)
                    return await self._task_checkpoint_exit(
                        current,
                        existing_task.id,
                    )
                except Exception:
                    current = await self._runtime.inspect(operation_id)
                    current_task = next(
                        task for task in current.tasks if task.id == existing_task.id
                    )
                    if current_task.status in {
                        TaskStatus.CLAIMED,
                        TaskStatus.RUNNING,
                        TaskStatus.MANUAL_RECOVERY_REQUIRED,
                    }:
                        return await self._task_checkpoint_exit(
                            current,
                            current_task.id,
                        )
                    return await self._runtime.fail(
                        operation_id,
                        "action_processing_failed",
                    )

                if resumed_evidence is None:
                    current = await self._runtime.inspect(operation_id)
                    return await self._task_checkpoint_exit(
                        current,
                        existing_task.id,
                    )
                evidence = resumed_evidence

            completed = await self._runtime.inspect(operation_id)
            completed_task = next(
                task for task in completed.tasks if task.id == evidence.task_id
            )
            evidence_by_id = {item.id: item for item in completed.evidence}
            observed_evidence_ids = {
                observation.evidence_id
                for observation in completed.observations
                if observation.task_id == completed_task.id
                and observation.evidence_id is not None
            }
            for evidence_id in completed_task.evidence_ids:
                if evidence_id in observed_evidence_ids:
                    continue
                accepted_evidence = evidence_by_id[evidence_id]
                terminal = await self._wall_budget_exit(
                    operation_id,
                    call_id=call.id,
                    task_id=completed_task.id,
                )
                if terminal is not None:
                    return terminal
                try:
                    observation = await self._await_with_wall_time(
                        operation_id,
                        lambda: self._domain.project_observation(accepted_evidence),
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
                terminal = await self._after_task_observation(
                    operation_id,
                    model_call.turn_id,
                    call.id,
                    completed_task.id,
                )
                if terminal is not None:
                    return terminal
        return None

    async def _after_action_rejection(
        self,
        operation_id: str,
        turn_id: str,
        call_id: str,
    ) -> LoopExit | None:
        terminal = await self._wall_budget_exit(operation_id, call_id=call_id)
        if terminal is not None:
            return terminal
        repaired = await self._runtime.inspect(operation_id)
        if (
            repaired.loop_state.identical_failure_count
            >= repaired.budgets.max_identical_failures
        ):
            return await self._runtime.fail_no_progress(operation_id, call_id)
        if repaired.loop_state.repair_count > repaired.budgets.max_repairs:
            return await self._runtime.fail_budget(
                operation_id,
                "repair_budget_exhausted",
                budget="repairs",
                limit=repaired.budgets.max_repairs,
                used=repaired.loop_state.repair_count,
                turn_id=turn_id,
                call_id=call_id,
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
                turn_id=turn_id,
                call_id=call_id,
            )
        return None

    async def _after_task_observation(
        self,
        operation_id: str,
        turn_id: str,
        call_id: str,
        task_id: str,
    ) -> LoopExit | None:
        terminal = await self._wall_budget_exit(
            operation_id,
            call_id=call_id,
            task_id=task_id,
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
                turn_id=turn_id,
                call_id=call_id,
                task_id=task_id,
            )
        return None

    async def _task_checkpoint_exit(
        self,
        snapshot: OperationSnapshot,
        task_id: str,
        *,
        outcome_unknown: bool = False,
    ) -> LoopExit:
        task = next(task for task in snapshot.tasks if task.id == task_id)
        if task.status in {TaskStatus.CLAIMED, TaskStatus.RUNNING}:
            has_live_lease = any(
                lease.task_id == task.id and lease.released_at is None
                for lease in snapshot.task_leases
            )
            if not has_live_lease:
                return await self._runtime.fail(
                    snapshot.operation.id,
                    "checkpoint_recovery_failed",
                )
            outcome_unknown = outcome_unknown or any(
                event.type == "task.outcome_unknown" and event.task_id == task.id
                for event in snapshot.events
            )
            return LoopExit(
                operation_id=snapshot.operation.id,
                kind=LoopExitKind.WAITING,
                reason=(
                    "task_outcome_unknown" if outcome_unknown else "task_lease_active"
                ),
                created_at=snapshot.operation.updated_at,
            )
        waiting_reason = {
            TaskStatus.WAITING_FOR_APPROVAL: "waiting_for_approval",
            TaskStatus.MANUAL_RECOVERY_REQUIRED: "manual_recovery_required",
        }.get(task.status)
        if waiting_reason is not None:
            return LoopExit(
                operation_id=snapshot.operation.id,
                kind=LoopExitKind.WAITING,
                reason=waiting_reason,
                created_at=snapshot.operation.updated_at,
            )
        if task.status is TaskStatus.CANCELLED:
            return await self._runtime.interrupt(
                snapshot.operation.id,
                "task_cancelled",
            )
        return await self._runtime.fail(
            snapshot.operation.id,
            "action_processing_failed",
        )

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
        async def persist_or_accept_terminal_race() -> None:
            try:
                await self._runtime.interrupt(operation_id)
            except OperationStateError:
                snapshot = await self._runtime.inspect(operation_id)
                if snapshot.operation.status not in {
                    OperationStatus.SUCCEEDED,
                    OperationStatus.FAILED,
                    OperationStatus.CANCELLED,
                    OperationStatus.INTERRUPTED,
                }:
                    raise

        commit = asyncio.create_task(persist_or_accept_terminal_race())
        while not commit.done():
            try:
                await asyncio.shield(commit)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        commit.result()


def _provider_supports_request_policy(
    provider: ModelProvider,
    request: ModelRequest,
) -> bool:
    check = getattr(provider, "supports_request_policy", None)
    if not callable(check):
        return request.allow_parallel_tool_calls is None
    try:
        supported = check(request)
    except Exception:
        return False
    return supported is True
