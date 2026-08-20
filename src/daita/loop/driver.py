"""A direct model -> tools -> model agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol, TypeVar

from .._json import FrozenJsonObject, canonical_json
from ..artifacts.models import (
    ArtifactDeliveryReceipt,
    ArtifactRef,
    artifact_delivery_receipt_from_mapping,
    artifact_ref_from_mapping,
)
from ..llm.errors import (
    ContextEvidencePressureExceeded,
    ContextWindowExceeded,
    ModelProviderError,
    ProviderErrorCode,
    RequestSensitivityUnavailable,
    ToolSurfaceLimitExceeded,
)
from ..llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelSensitivity,
    ModelStreamCompleted,
    ModelTextDelta,
    ModelToolCallDelta,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from ..llm.pricing import (
    CostEstimate,
    CostEstimateStatus,
    aggregate_cost_estimates,
    canonical_decimal,
    format_cost_estimate,
)
from ..llm.protocols import (
    ModelProvider,
    StreamingModelProvider,
    provider_has_complete_pricing,
    provider_supports_request_policy,
)
from ..observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely
from .models import (
    LoopExit,
    LoopExitKind,
    LoopLimits,
    RunInput,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolBatchOutcome,
    Transcript,
    validate_completed_transcript,
)

_T = TypeVar("_T")


def _utc_now() -> datetime:
    return datetime.now(UTC)


class ContextBuilder(Protocol):
    async def prepare(
        self,
        run: RunInput,
        messages: tuple[CanonicalMessage, ...],
        tools: tuple[ToolDefinition, ...],
    ) -> object: ...

    def project(
        self,
        snapshot: object,
        messages: tuple[CanonicalMessage, ...],
        *,
        step: int,
        final: bool = False,
        previous_request_input_tokens: int | None = None,
    ) -> ModelRequest: ...


class ToolRuntime(Protocol):
    async def definitions(self, run: RunInput) -> tuple[ToolDefinition, ...]: ...

    async def execute_all(
        self,
        run: RunInput,
        calls: tuple[ToolCall, ...],
        *,
        sensitivity: ModelSensitivity,
    ) -> ToolBatchOutcome:
        """Return one ordered, cancellation-safe outcome for the whole batch."""

        ...


class TranscriptStore(Protocol):
    async def start(self, run: RunInput) -> Transcript: ...

    async def append(self, run_id: str, message: CanonicalMessage) -> None: ...

    async def complete(
        self,
        result: LoopExit,
        final_message: CanonicalMessage,
    ) -> None: ...

    async def finish(self, result: LoopExit) -> None: ...


class InMemoryTranscriptStore:
    """Small default store for embedded use and focused loop tests."""

    def __init__(self) -> None:
        self._transcripts: dict[str, Transcript] = {}
        self._results: dict[str, LoopExit] = {}

    async def start(self, run: RunInput) -> Transcript:
        if run.id in self._transcripts:
            raise ValueError(f"run already exists: {run.id}")
        transcript = Transcript(run=run)
        self._transcripts[run.id] = transcript
        return transcript

    async def append(self, run_id: str, message: CanonicalMessage) -> None:
        try:
            current = self._transcripts[run_id]
        except KeyError as error:
            raise KeyError(f"unknown run: {run_id}") from error
        self._transcripts[run_id] = Transcript(
            run=current.run,
            messages=(*current.messages, message),
        )

    async def finish(self, result: LoopExit) -> None:
        if result.run_id not in self._transcripts:
            raise KeyError(f"unknown run: {result.run_id}")
        if result.kind is LoopExitKind.COMPLETED:
            raise ValueError("completed runs require atomic transcript completion")
        self._results[result.run_id] = result

    async def complete(
        self,
        result: LoopExit,
        final_message: CanonicalMessage,
    ) -> None:
        try:
            current = self._transcripts[result.run_id]
        except KeyError as error:
            raise KeyError(f"unknown run: {result.run_id}") from error
        candidate = Transcript(
            run=current.run,
            messages=(*current.messages, final_message),
        )
        validate_completed_transcript(candidate, result)
        self._transcripts[result.run_id] = candidate
        self._results[result.run_id] = result

    async def load(self, run_id: str) -> Transcript:
        try:
            return self._transcripts[run_id]
        except KeyError as error:
            raise KeyError(f"unknown run: {run_id}") from error

    async def result(self, run_id: str) -> LoopExit | None:
        return self._results.get(run_id)


class AgentLoop:
    """Own only cognitive progression; tools and persistence keep their own data."""

    def __init__(
        self,
        *,
        model: ModelProvider,
        context_builder: ContextBuilder,
        tools: ToolRuntime,
        transcripts: TranscriptStore | None = None,
        limits: LoopLimits = LoopLimits(),
        clock: Callable[[], datetime] = _utc_now,
        observer: AgentObserver | None = None,
        stream_model_calls: bool = False,
    ) -> None:
        if not isinstance(limits, LoopLimits):
            raise TypeError("limits must be LoopLimits")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        if not isinstance(stream_model_calls, bool):
            raise TypeError("stream_model_calls must be a boolean")
        self._model = model
        self._context_builder = context_builder
        self._tools = tools
        self._transcripts = transcripts or InMemoryTranscriptStore()
        self._limits = limits
        self._clock = clock
        self._observer = observer
        self._stream_model_calls = stream_model_calls

    async def run(
        self,
        run: RunInput,
        *,
        prior_messages: tuple[CanonicalMessage, ...] = (),
    ) -> LoopExit:
        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        if any(not isinstance(message, CanonicalMessage) for message in prior_messages):
            raise TypeError("prior_messages must contain CanonicalMessage records")
        if run.conversation_id is None:
            run = replace(run, conversation_id=run.id)
        transcript = await self._transcripts.start(run)
        run_started = asyncio.get_running_loop().time()
        if self._observer is not None:
            self._emit(
                AgentEventKind.RUN_STARTED,
                run,
                {"agent_id": run.agent_id},
            )
        messages = (*prior_messages, *transcript.messages)
        current_start = len(prior_messages)
        usage = ModelUsage(cost_estimate=CostEstimate.unavailable("no_model_attempts"))
        artifacts: list[ArtifactRef] = []
        artifact_deliveries: list[ArtifactDeliveryReceipt] = []
        previous_request_input_tokens: int | None = None
        tool_call_count = 0
        run_route: object | None = None

        try:
            user = CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(run.message),),
            )
            await self._transcripts.append(run.id, user)
            messages = (*messages, user)
            started = asyncio.get_running_loop().time()
            deadline = started + self._limits.max_wall_time_seconds
            definitions = _bounded_tool_definitions(
                await _before(deadline, self._tools.definitions(run)),
                self._limits,
            )
            context_snapshot = await _before(
                deadline,
                self._context_builder.prepare(run, messages, definitions),
            )
            for step in range(1, self._limits.max_steps + 1):
                if self._wall_time_exhausted(started):
                    return await self._finish(
                        run,
                        LoopExitKind.FAILED,
                        "wall_time_exhausted",
                        step - 1,
                        usage,
                        run_started,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )
                request = self._context_builder.project(
                    context_snapshot,
                    messages[current_start:],
                    step=step,
                    previous_request_input_tokens=previous_request_input_tokens,
                )
                if run_route is None:
                    try:
                        run_route = _begin_run_route(self._model, request)
                    except ModelProviderError:
                        return await self._finish(
                            run,
                            LoopExitKind.FAILED,
                            "model_route_ineligible",
                            step - 1,
                            usage,
                            run_started,
                            artifacts=tuple(artifacts),
                            artifact_deliveries=tuple(artifact_deliveries),
                        )
                if not _provider_supports_run_request(
                    self._model,
                    run_route,
                    request,
                ):
                    return await self._finish(
                        run,
                        LoopExitKind.FAILED,
                        "model_route_ineligible",
                        step - 1,
                        usage,
                        run_started,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )
                if not self._cost_limit_allows_request(request, run_route):
                    return await self._finish(
                        run,
                        LoopExitKind.FAILED,
                        "cost_limit_unpriced_route",
                        step - 1,
                        usage,
                        run_started,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )
                model_started = (
                    asyncio.get_running_loop().time()
                    if self._observer is not None
                    else None
                )
                response = await self._model_response(
                    request,
                    run,
                    model_call_index=step,
                    deadline=deadline,
                    run_route=run_route,
                )
                model_duration_ms = (
                    _duration_ms(model_started) if model_started is not None else None
                )
                usage = _add_usage(usage, response.usage)
                previous_request_input_tokens = response.request_input_tokens
                assistant = _assistant_message(response)
                if self._observer is not None:
                    assert model_duration_ms is not None
                    self._emit_model_completed(
                        run,
                        response,
                        model_duration_ms,
                        model_call_index=step,
                    )

                if response.finish_reason is FinishReason.STOP:
                    assert response.text is not None and not response.tool_calls
                    return await self._finish(
                        run,
                        LoopExitKind.COMPLETED,
                        "completed",
                        step,
                        usage,
                        run_started,
                        final_text=response.text,
                        final_message=assistant,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )

                if response.finish_reason is not FinishReason.TOOL_CALLS:
                    await self._transcripts.append(run.id, assistant)
                    messages = (*messages, assistant)
                    return await self._finish(
                        run,
                        LoopExitKind.FAILED,
                        _finish_reason_failure(response.finish_reason),
                        step,
                        usage,
                        run_started,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )

                assert response.tool_calls
                if len(response.tool_calls) > self._limits.max_tool_calls_per_response:
                    return await self._finish(
                        run,
                        LoopExitKind.FAILED,
                        "tool_calls_per_response_exceeded",
                        step,
                        usage,
                        run_started,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )
                if (
                    tool_call_count + len(response.tool_calls)
                    > self._limits.max_tool_calls_per_run
                ):
                    return await self._finish(
                        run,
                        LoopExitKind.FAILED,
                        "tool_calls_per_run_exceeded",
                        step,
                        usage,
                        run_started,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )
                tool_call_count += len(response.tool_calls)
                await self._transcripts.append(run.id, assistant)
                messages = (*messages, assistant)

                outcome, cancellation_requested = await _tool_batch_before(
                    deadline,
                    self._tools.execute_all(
                        run,
                        response.tool_calls,
                        sensitivity=request.sensitivity,
                    ),
                    response.tool_calls,
                    recovery_timeout_seconds=(
                        self._limits.side_effect_recovery_timeout_seconds + 0.25
                    ),
                )
                results = outcome.ordered_results
                if len(results) != len(response.tool_calls) or any(
                    result.call_id != call.id
                    for call, result in zip(response.tool_calls, results, strict=True)
                ):
                    raise ValueError(
                        "tool runtime must return one ordered result per tool call"
                    )
                for result in results:
                    tool_message = CanonicalMessage(
                        role=MessageRole.TOOL,
                        content=(result,),
                    )
                    if outcome.interruption_kind is None:
                        await self._transcripts.append(run.id, tool_message)
                    else:
                        await _complete_before_cancellation(
                            self._transcripts.append(run.id, tool_message)
                        )
                    messages = (*messages, tool_message)
                    artifact = _artifact_ref(result)
                    if artifact is not None:
                        artifacts.append(artifact)
                    receipt = _artifact_delivery(result)
                    if receipt is not None:
                        artifact_deliveries.append(receipt)

                if outcome.interruption_kind is not None:
                    if cancellation_requested:
                        raise asyncio.CancelledError
                    if outcome.interruption_kind is ToolBatchInterruption.DEADLINE:
                        return await self._finish(
                            run,
                            LoopExitKind.FAILED,
                            "wall_time_exhausted",
                            step,
                            usage,
                            run_started,
                            artifacts=tuple(artifacts),
                            artifact_deliveries=tuple(artifact_deliveries),
                        )
                    return await self._finish(
                        run,
                        LoopExitKind.INTERRUPTED,
                        "tool_batch_interrupted",
                        step,
                        usage,
                        run_started,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )

                budget_reason = self._usage_limit_reason(usage)
                if budget_reason is not None:
                    if budget_reason.startswith("cost_limit_"):
                        return await self._finish(
                            run,
                            LoopExitKind.FAILED,
                            budget_reason,
                            step,
                            usage,
                            run_started,
                            artifacts=tuple(artifacts),
                            artifact_deliveries=tuple(artifact_deliveries),
                        )
                    return await self._wrap_up(
                        run,
                        messages[current_start:],
                        step,
                        usage,
                        budget_reason,
                        deadline,
                        run_started,
                        context_snapshot,
                        previous_request_input_tokens,
                        run_route,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )

            return await self._wrap_up(
                run,
                messages[current_start:],
                self._limits.max_steps,
                usage,
                "step_limit_reached",
                deadline,
                run_started,
                context_snapshot,
                previous_request_input_tokens,
                run_route,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
        except asyncio.CancelledError:
            await self._finish_best_effort(
                run,
                LoopExitKind.INTERRUPTED,
                "cancelled",
                _completed_steps(messages[current_start:]),
                usage,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
            raise
        except TimeoutError:
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                "wall_time_exhausted",
                _completed_steps(messages[current_start:]),
                usage,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
        except ContextWindowExceeded:
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                "context_window_exceeded",
                _completed_steps(messages[current_start:]),
                usage,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
        except ContextEvidencePressureExceeded:
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                "context_evidence_limit_exceeded",
                _completed_steps(messages[current_start:]),
                usage,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
        except ToolSurfaceLimitExceeded:
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                "tool_surface_limit_exceeded",
                _completed_steps(messages[current_start:]),
                usage,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
        except RequestSensitivityUnavailable:
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                "request_sensitivity_unavailable",
                _completed_steps(messages[current_start:]),
                usage,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
        except ModelProviderError as error:
            usage = _add_usage(usage, error.usage)
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                error.code.value,
                _completed_steps(messages[current_start:]),
                usage,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
        except Exception:
            await self._finish_best_effort(
                run,
                LoopExitKind.FAILED,
                "unexpected_internal_error",
                _completed_steps(messages[current_start:]),
                usage,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
            raise

    async def _wrap_up(
        self,
        run: RunInput,
        messages: tuple[CanonicalMessage, ...],
        steps: int,
        usage: ModelUsage,
        reason: str,
        deadline: float,
        run_started: float,
        context_snapshot: object,
        previous_request_input_tokens: int | None,
        run_route: object | None,
        *,
        artifacts: tuple[ArtifactRef, ...],
        artifact_deliveries: tuple[ArtifactDeliveryReceipt, ...],
    ) -> LoopExit:
        if asyncio.get_running_loop().time() >= deadline:
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                reason,
                steps,
                usage,
                run_started,
                artifacts=artifacts,
                artifact_deliveries=artifact_deliveries,
            )
        request = self._context_builder.project(
            context_snapshot,
            messages,
            step=steps + 1,
            final=True,
            previous_request_input_tokens=previous_request_input_tokens,
        )
        if not _provider_supports_run_request(self._model, run_route, request):
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                "model_route_ineligible",
                steps,
                usage,
                run_started,
                artifacts=artifacts,
                artifact_deliveries=artifact_deliveries,
            )
        if not self._cost_limit_allows_request(request, run_route):
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                "cost_limit_unpriced_route",
                steps,
                usage,
                run_started,
                artifacts=artifacts,
                artifact_deliveries=artifact_deliveries,
            )
        try:
            model_started = (
                asyncio.get_running_loop().time()
                if self._observer is not None
                else None
            )
            response = await self._model_response(
                request,
                run,
                model_call_index=steps + 1,
                deadline=deadline,
                run_route=run_route,
            )
        except ModelProviderError as error:
            usage = _add_usage(usage, error.usage)
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                reason,
                steps,
                usage,
                run_started,
                artifacts=artifacts,
                artifact_deliveries=artifact_deliveries,
            )
        model_duration_ms = (
            _duration_ms(model_started) if model_started is not None else None
        )
        usage = _add_usage(usage, response.usage)
        if response.finish_reason is FinishReason.TOOL_CALLS:
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                "tool_free_wrap_up_returned_tool_calls",
                steps,
                usage,
                run_started,
                artifacts=artifacts,
                artifact_deliveries=artifact_deliveries,
            )
        assistant = _assistant_message(response)
        if self._observer is not None:
            assert model_duration_ms is not None
            self._emit_model_completed(
                run,
                response,
                model_duration_ms,
                model_call_index=steps + 1,
            )
        if response.finish_reason is not FinishReason.STOP:
            await self._transcripts.append(run.id, assistant)
            return await self._finish(
                run,
                LoopExitKind.FAILED,
                _finish_reason_failure(response.finish_reason),
                steps,
                usage,
                run_started,
                artifacts=artifacts,
                artifact_deliveries=artifact_deliveries,
            )
        assert response.text is not None and not response.tool_calls
        return await self._finish(
            run,
            LoopExitKind.COMPLETED,
            reason,
            steps,
            usage,
            run_started,
            final_text=response.text,
            final_message=assistant,
            artifacts=artifacts,
            artifact_deliveries=artifact_deliveries,
        )

    async def _finish(
        self,
        run: RunInput,
        kind: LoopExitKind,
        reason: str,
        steps: int,
        usage: ModelUsage,
        run_started: float,
        *,
        final_text: str | None = None,
        final_message: CanonicalMessage | None = None,
        artifacts: tuple[ArtifactRef, ...] = (),
        artifact_deliveries: tuple[ArtifactDeliveryReceipt, ...] = (),
    ) -> LoopExit:
        result = LoopExit(
            run_id=run.id,
            conversation_id=run.conversation_id or run.id,
            kind=kind,
            reason=reason,
            final_text=final_text,
            steps=steps,
            usage=usage,
            artifacts=artifacts,
            artifact_deliveries=artifact_deliveries,
            created_at=self._clock(),
        )
        if kind is LoopExitKind.COMPLETED:
            if final_message is None:
                raise ValueError("completed loop exit requires a final assistant")
            await self._transcripts.complete(result, final_message)
        else:
            if final_message is not None:
                raise ValueError("only completed loop exits accept a final assistant")
            await self._transcripts.finish(result)
        if self._observer is not None:
            self._emit(
                AgentEventKind.RUN_COMPLETED,
                run,
                {
                    "exit_kind": result.kind.value,
                    "reason": result.reason,
                    "steps": result.steps,
                    "duration_ms": _duration_ms(run_started),
                    "input_tokens": result.usage.input_tokens,
                    "output_tokens": result.usage.output_tokens,
                    "reasoning_tokens": result.usage.reasoning_tokens,
                    "cache_read_tokens": result.usage.cache_read_tokens,
                    "cache_write_tokens": result.usage.cache_write_tokens,
                    "total_tokens": result.usage.total_tokens,
                    "cost_status": result.usage.cost_estimate.status.value,
                    "cost_amount_usd": (
                        None
                        if result.usage.cost_estimate.amount_usd is None
                        else canonical_decimal(result.usage.cost_estimate.amount_usd)
                    ),
                    "cost_basis": (
                        None
                        if result.usage.cost_estimate.basis is None
                        else result.usage.cost_estimate.basis.value
                    ),
                    "cost_rate_schedule_id": (
                        result.usage.cost_estimate.rate_schedule_id
                    ),
                    "cost_code": result.usage.cost_estimate.code,
                    "cost_display": format_cost_estimate(result.usage.cost_estimate),
                },
            )
        return result

    async def _finish_best_effort(
        self,
        run: RunInput,
        kind: LoopExitKind,
        reason: str,
        steps: int,
        usage: ModelUsage,
        run_started: float,
        *,
        artifacts: tuple[ArtifactRef, ...] = (),
        artifact_deliveries: tuple[ArtifactDeliveryReceipt, ...] = (),
    ) -> None:
        try:
            await _complete_before_cancellation(
                self._finish(
                    run,
                    kind,
                    reason,
                    steps,
                    usage,
                    run_started,
                    artifacts=artifacts,
                    artifact_deliveries=artifact_deliveries,
                )
            )
        except BaseException:
            pass

    def _emit_model_completed(
        self,
        run: RunInput,
        response: ModelResponse,
        duration_ms: int,
        *,
        model_call_index: int,
    ) -> None:
        self._emit(
            AgentEventKind.MODEL_COMPLETED,
            run,
            {
                "provider_id": response.provider_id or self._model.provider_id,
                "model_call_index": model_call_index,
                "has_text": response.text is not None,
                "has_tool_calls": bool(response.tool_calls),
                "duration_ms": duration_ms,
                "input_tokens": response.usage.input_tokens,
                "context_input_tokens": response.request_input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
        )

    async def _model_response(
        self,
        request: ModelRequest,
        run: RunInput,
        *,
        model_call_index: int,
        deadline: float,
        run_route: object | None,
    ) -> ModelResponse:
        model = self._model
        if not self._stream_model_calls or not isinstance(
            model, StreamingModelProvider
        ):
            generate_for_run = getattr(model, "generate_for_run", None)
            awaitable = (
                generate_for_run(run_route, request)
                if run_route is not None and callable(generate_for_run)
                else model.generate(request)
            )
            return await _before(deadline, awaitable)

        async def consume() -> ModelResponse:
            completed: ModelResponse | None = None
            stream_for_run = getattr(model, "stream_for_run", None)
            events = (
                stream_for_run(run_route, request)
                if run_route is not None and callable(stream_for_run)
                else model.stream(request)
            )
            async for event in events:
                if completed is not None:
                    raise ModelProviderError(
                        ProviderErrorCode.MALFORMED_RESPONSE,
                        "model stream continued after its canonical completion",
                        provider_id=self._model.provider_id,
                    )
                if isinstance(event, ModelTextDelta):
                    self._emit_model_text_delta(
                        run,
                        event.text,
                        model_call_index=model_call_index,
                    )
                elif isinstance(event, ModelToolCallDelta):
                    pass
                elif isinstance(event, ModelStreamCompleted):
                    completed = event.response
                else:
                    raise ModelProviderError(
                        ProviderErrorCode.MALFORMED_RESPONSE,
                        "model stream returned an unsupported canonical event",
                        provider_id=self._model.provider_id,
                    )
                # A deterministic or local provider may have an immediately-ready
                # iterator. Yield so presentation, input, and cancellation remain
                # responsive even for a high-rate canonical stream.
                await asyncio.sleep(0)
            if completed is None:
                raise ModelProviderError(
                    ProviderErrorCode.MALFORMED_RESPONSE,
                    "model stream ended without its canonical completion",
                    provider_id=self._model.provider_id,
                )
            return completed

        return await _before(deadline, consume())

    def _emit_model_text_delta(
        self,
        run: RunInput,
        text: str,
        *,
        model_call_index: int,
    ) -> None:
        if self._observer is None:
            return
        for offset in range(0, len(text), 1_024):
            self._emit(
                AgentEventKind.MODEL_TEXT_DELTA,
                run,
                {
                    "model_call_index": model_call_index,
                    "text": text[offset : offset + 1_024],
                },
            )

    def _emit(
        self,
        kind: AgentEventKind,
        run: RunInput,
        data: dict[str, object],
    ) -> None:
        try:
            event = AgentEvent(
                kind=kind,
                occurred_at=self._clock(),
                run_id=run.id,
                conversation_id=run.conversation_id or run.id,
                data=FrozenJsonObject.from_mapping(data),
            )
        except Exception:
            return
        _emit_safely(
            self._observer,
            event,
        )

    def _wall_time_exhausted(self, started: float) -> bool:
        return (
            asyncio.get_running_loop().time() - started
            >= self._limits.max_wall_time_seconds
        )

    def _cost_limit_allows_request(
        self,
        request: ModelRequest,
        run_route: object | None,
    ) -> bool:
        if self._limits.max_estimated_cost_usd is None:
            return True
        return _provider_has_complete_run_pricing(
            self._model,
            run_route,
            request,
        )

    def _usage_limit_reason(self, usage: ModelUsage) -> str | None:
        if usage.total_tokens >= self._limits.max_total_tokens:
            return "token_limit_reached"
        cost_limit = self._limits.max_estimated_cost_usd
        if cost_limit is not None:
            estimate = usage.cost_estimate
            if estimate.status is not CostEstimateStatus.COMPLETE:
                return "cost_limit_unpriced_route"
            assert estimate.amount_usd is not None
            if estimate.amount_usd >= cost_limit:
                return "cost_limit_reached"
        return None


def _bounded_tool_definitions(
    definitions: tuple[ToolDefinition, ...],
    limits: LoopLimits,
) -> tuple[ToolDefinition, ...]:
    definitions = tuple(definitions)
    if any(not isinstance(item, ToolDefinition) for item in definitions):
        raise TypeError("tool runtime definitions must contain ToolDefinition records")
    definition_bytes = len(
        canonical_json(
            [
                {
                    "name": item.name,
                    "description": item.description,
                    "input_schema": item.input_schema,
                }
                for item in definitions
            ]
        ).encode("utf-8")
    )
    if (
        len(definitions) > limits.max_projected_tools
        or definition_bytes > limits.max_projected_tool_definition_bytes
    ):
        raise ToolSurfaceLimitExceeded(
            observed_tools=len(definitions),
            maximum_tools=limits.max_projected_tools,
            observed_definition_bytes=definition_bytes,
            maximum_definition_bytes=limits.max_projected_tool_definition_bytes,
        )
    return definitions


def _assistant_message(response: ModelResponse) -> CanonicalMessage:
    content = () if response.text is None else (TextBlock(response.text),)
    return CanonicalMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        tool_calls=response.tool_calls,
        provider_id=response.provider_id,
        provider_metadata=response.provider_metadata,
    )


def _finish_reason_failure(finish_reason: FinishReason) -> str:
    failures = {
        FinishReason.LENGTH: "model_output_limit",
        FinishReason.CONTENT_FILTER: "content_filtered",
        FinishReason.ERROR: "model_response_error",
    }
    try:
        return failures[finish_reason]
    except KeyError as error:
        raise ValueError(
            f"finish reason is not a terminal failure: {finish_reason.value}"
        ) from error


def _begin_run_route(provider: object, request: ModelRequest) -> object | None:
    begin = getattr(provider, "begin_run", None)
    if not callable(begin):
        return None
    return begin(request.sensitivity)


def _provider_supports_run_request(
    provider: object,
    run_route: object | None,
    request: ModelRequest,
) -> bool:
    probe = getattr(provider, "supports_run_request", None)
    if run_route is not None and callable(probe):
        try:
            return probe(run_route, request) is True
        except Exception:
            return False
    return provider_supports_request_policy(provider, request)


def _provider_has_complete_run_pricing(
    provider: object,
    run_route: object | None,
    request: ModelRequest,
) -> bool:
    probe = getattr(provider, "has_complete_run_pricing", None)
    if run_route is not None and callable(probe):
        try:
            return probe(run_route, request) is True
        except Exception:
            return False
    return provider_has_complete_pricing(provider, request)


def _add_usage(left: ModelUsage, right: ModelUsage) -> ModelUsage:
    if (
        left.total_tokens == 0
        and left.reasoning_tokens == 0
        and left.cache_read_tokens == 0
        and left.cache_write_tokens == 0
        and left.cost_estimate.code == "no_model_attempts"
    ):
        return right
    return ModelUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
        cache_read_tokens=left.cache_read_tokens + right.cache_read_tokens,
        cache_write_tokens=left.cache_write_tokens + right.cache_write_tokens,
        cost_estimate=aggregate_cost_estimates(
            (left.cost_estimate, right.cost_estimate)
        ),
    )


def _completed_steps(messages: tuple[CanonicalMessage, ...]) -> int:
    return sum(message.role is MessageRole.ASSISTANT for message in messages)


def _artifact_ref(result: ToolResultBlock) -> ArtifactRef | None:
    if result.is_error:
        return None
    value = result.output.get("artifact")
    if not isinstance(value, Mapping):
        return None
    try:
        ref = artifact_ref_from_mapping(value)
    except (TypeError, ValueError):
        return None
    return ref if ref.call_id == result.call_id else None


def _artifact_delivery(result: ToolResultBlock) -> ArtifactDeliveryReceipt | None:
    if result.is_error or result.output.get("kind") != "artifact.delivery_receipt":
        return None
    value = result.output.get("data")
    if not isinstance(value, Mapping):
        return None
    try:
        return artifact_delivery_receipt_from_mapping(value)
    except (TypeError, ValueError):
        return None


def _duration_ms(started: float) -> int:
    elapsed = asyncio.get_running_loop().time() - started
    return max(0, int(elapsed * 1_000))


async def _before(deadline: float, awaitable: Awaitable[_T]) -> _T:
    async with asyncio.timeout_at(deadline):
        return await awaitable


async def _tool_batch_before(
    deadline: float,
    awaitable: Awaitable[ToolBatchOutcome],
    calls: tuple[ToolCall, ...],
    *,
    recovery_timeout_seconds: float,
) -> tuple[ToolBatchOutcome, bool]:
    worker: asyncio.Future[ToolBatchOutcome] = asyncio.ensure_future(awaitable)
    remaining = max(0.0, deadline - asyncio.get_running_loop().time())
    try:
        done, _pending = await asyncio.wait((worker,), timeout=remaining)
    except asyncio.CancelledError:
        worker.cancel(ToolBatchInterruption.CANCELLED.value)
        outcome = await _recover_tool_batch(
            worker,
            calls,
            ToolBatchInterruption.CANCELLED,
            timeout_seconds=recovery_timeout_seconds,
        )
        return outcome, True
    if done:
        return _validated_tool_batch(worker.result()), False
    worker.cancel(ToolBatchInterruption.DEADLINE.value)
    outcome = await _recover_tool_batch(
        worker,
        calls,
        ToolBatchInterruption.DEADLINE,
        timeout_seconds=recovery_timeout_seconds,
    )
    return outcome, False


async def _recover_tool_batch(
    worker: asyncio.Future[ToolBatchOutcome],
    calls: tuple[ToolCall, ...],
    interruption: ToolBatchInterruption,
    *,
    timeout_seconds: float,
) -> ToolBatchOutcome:
    settled = await _complete_before_cancellation(
        asyncio.wait((worker,), timeout=timeout_seconds)
    )
    if not settled[0]:
        worker.cancel(interruption.value)
        worker.add_done_callback(_consume_tool_batch_future)
        return _unknown_batch(calls, interruption)
    try:
        outcome = await _complete_before_cancellation(worker)
    except asyncio.CancelledError:
        return _unknown_batch(calls, interruption)
    return _validated_tool_batch(outcome, interruption=interruption)


def _consume_tool_batch_future(
    worker: asyncio.Future[ToolBatchOutcome],
) -> None:
    try:
        worker.exception()
    except BaseException:
        pass


def _validated_tool_batch(
    value: ToolBatchOutcome,
    *,
    interruption: ToolBatchInterruption | None = None,
) -> ToolBatchOutcome:
    if not isinstance(value, ToolBatchOutcome):
        raise TypeError("tool runtime must return ToolBatchOutcome")
    if interruption is None or value.interruption_kind is interruption:
        return value
    return ToolBatchOutcome(
        ordered_results=value.ordered_results,
        interruption_kind=interruption,
        outcome_certainty=value.outcome_certainty,
    )


def _unknown_batch(
    calls: tuple[ToolCall, ...],
    interruption: ToolBatchInterruption,
) -> ToolBatchOutcome:
    return ToolBatchOutcome(
        ordered_results=tuple(
            ToolResultBlock(
                call_id=call.id,
                is_error=True,
                output={
                    "error": {
                        "code": "outcome_unknown",
                        "message": (
                            "The tool runtime did not return a classified outcome "
                            "within the bounded interruption path."
                        ),
                        "details": {
                            "interruption_kind": interruption.value,
                            "outcome_certainty": (
                                ToolBatchCertainty.OUTCOME_UNKNOWN.value
                            ),
                        },
                    }
                },
            )
            for call in calls
        ),
        interruption_kind=interruption,
        outcome_certainty=ToolBatchCertainty.OUTCOME_UNKNOWN,
    )


async def _complete_before_cancellation(awaitable: Awaitable[_T]) -> _T:
    async def complete() -> _T:
        return await awaitable

    worker = asyncio.create_task(complete())
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
    return worker.result()


__all__ = [
    "AgentLoop",
    "ContextBuilder",
    "InMemoryTranscriptStore",
    "ToolRuntime",
    "TranscriptStore",
]
