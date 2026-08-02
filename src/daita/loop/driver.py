"""A direct model -> tools -> model agent loop."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from typing import Protocol, TypeVar

from .._json import FrozenJsonObject
from ..artifacts.models import (
    ArtifactDeliveryReceipt,
    ArtifactRef,
    artifact_delivery_receipt_from_mapping,
    artifact_ref_from_mapping,
)
from ..llm.errors import ContextWindowExceeded, ModelProviderError
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelRequest,
    ModelResponse,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from ..llm.protocols import ModelProvider, provider_has_complete_pricing
from ..llm.pricing import (
    CostEstimate,
    CostEstimateStatus,
    aggregate_cost_estimates,
    canonical_decimal,
    format_cost_estimate,
)
from ..observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely
from .models import LoopExit, LoopExitKind, LoopLimits, RunInput, Transcript

_T = TypeVar("_T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ContextBuilder(Protocol):
    async def build(
        self,
        run: RunInput,
        messages: tuple[CanonicalMessage, ...],
        tools: tuple[ToolDefinition, ...],
        *,
        step: int,
        final: bool = False,
    ) -> ModelRequest: ...


class ToolRuntime(Protocol):
    async def definitions(self, run: RunInput) -> tuple[ToolDefinition, ...]: ...

    async def execute_all(
        self,
        run: RunInput,
        calls: tuple[ToolCall, ...],
    ) -> tuple[ToolResultBlock, ...]:
        """Return one result per call, in the original call order."""

        ...


class TranscriptStore(Protocol):
    async def start(self, run: RunInput) -> Transcript: ...

    async def append(self, run_id: str, message: CanonicalMessage) -> None: ...

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
    ) -> None:
        if not isinstance(limits, LoopLimits):
            raise TypeError("limits must be LoopLimits")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        self._model = model
        self._context_builder = context_builder
        self._tools = tools
        self._transcripts = transcripts or InMemoryTranscriptStore()
        self._limits = limits
        self._clock = clock
        self._observer = observer

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

        try:
            user = CanonicalMessage(
                role=MessageRole.USER,
                content=(TextBlock(run.message),),
            )
            await self._transcripts.append(run.id, user)
            messages = (*messages, user)
            started = asyncio.get_running_loop().time()
            deadline = started + self._limits.max_wall_time_seconds
            definitions = await _before(deadline, self._tools.definitions(run))
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
                request = await _before(
                    deadline,
                    self._context_builder.build(
                        run,
                        messages,
                        definitions,
                        step=step,
                    ),
                )
                if not self._cost_limit_allows_request(request):
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
                response = await _before(deadline, self._model.generate(request))
                model_duration_ms = (
                    _duration_ms(model_started) if model_started is not None else None
                )
                usage = _add_usage(usage, response.usage)
                assistant = _assistant_message(response)
                await self._transcripts.append(run.id, assistant)
                messages = (*messages, assistant)
                if self._observer is not None:
                    assert model_duration_ms is not None
                    self._emit_model_completed(run, response, model_duration_ms)

                if not response.tool_calls:
                    assert response.text is not None
                    return await self._finish(
                        run,
                        LoopExitKind.COMPLETED,
                        "completed",
                        step,
                        usage,
                        run_started,
                        final_text=response.text,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )

                results = await _before(
                    deadline,
                    self._tools.execute_all(run, response.tool_calls),
                )
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
                    await self._transcripts.append(run.id, tool_message)
                    messages = (*messages, tool_message)
                    artifact = _artifact_ref(result)
                    if artifact is not None:
                        artifacts.append(artifact)
                    receipt = _artifact_delivery(result)
                    if receipt is not None:
                        artifact_deliveries.append(receipt)

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
                        messages,
                        step,
                        usage,
                        budget_reason,
                        deadline,
                        run_started,
                        artifacts=tuple(artifacts),
                        artifact_deliveries=tuple(artifact_deliveries),
                    )

            return await self._wrap_up(
                run,
                messages,
                self._limits.max_steps,
                usage,
                "step_limit_reached",
                deadline,
                run_started,
                artifacts=tuple(artifacts),
                artifact_deliveries=tuple(artifact_deliveries),
            )
        except asyncio.CancelledError:
            await self._finish(
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

    async def _wrap_up(
        self,
        run: RunInput,
        messages: tuple[CanonicalMessage, ...],
        steps: int,
        usage: ModelUsage,
        reason: str,
        deadline: float,
        run_started: float,
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
        request = await _before(
            deadline,
            self._context_builder.build(
                run,
                messages,
                (),
                step=steps + 1,
                final=True,
            ),
        )
        if not self._cost_limit_allows_request(request):
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
            response = await _before(deadline, self._model.generate(request))
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
        assistant = _assistant_message(response)
        await self._transcripts.append(run.id, assistant)
        if self._observer is not None:
            assert model_duration_ms is not None
            self._emit_model_completed(run, response, model_duration_ms)
        if response.text is None:
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
        return await self._finish(
            run,
            LoopExitKind.COMPLETED,
            reason,
            steps,
            usage,
            run_started,
            final_text=response.text,
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

    def _emit_model_completed(
        self,
        run: RunInput,
        response: ModelResponse,
        duration_ms: int,
    ) -> None:
        self._emit(
            AgentEventKind.MODEL_COMPLETED,
            run,
            {
                "provider_id": response.provider_id or self._model.provider_id,
                "duration_ms": duration_ms,
                "input_tokens": response.usage.input_tokens,
                "context_input_tokens": response.request_input_tokens,
                "output_tokens": response.usage.output_tokens,
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

    def _cost_limit_allows_request(self, request: ModelRequest) -> bool:
        if self._limits.max_estimated_cost_usd is None:
            return True
        return provider_has_complete_pricing(self._model, request)

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


def _assistant_message(response: ModelResponse) -> CanonicalMessage:
    content = () if response.text is None else (TextBlock(response.text),)
    return CanonicalMessage(
        role=MessageRole.ASSISTANT,
        content=content,
        tool_calls=response.tool_calls,
        provider_id=response.provider_id,
        provider_metadata=response.provider_metadata,
    )


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


__all__ = [
    "AgentLoop",
    "ContextBuilder",
    "InMemoryTranscriptStore",
    "ToolRuntime",
    "TranscriptStore",
]
