"""One domain-neutral model-to-capability execution boundary."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, cast

from ._json import FrozenJsonObject, canonical_json
from .artifacts.models import (
    ArtifactDraft,
    ArtifactError,
    ArtifactRef,
    artifact_ref_to_mapping,
    canonical_artifact_filename,
)
from .artifacts.store import AgentHomeArtifactStore
from .capabilities import (
    AccessMode,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    CapabilityRegistry,
    Executor,
    SideEffectExecutor,
    ToolExecution,
    ToolOutput,
    ToolOutputValidationError,
    ToolView,
)
from .errors import DaitaError
from .llm.models import ToolCall, ToolDefinition, ToolResultBlock
from .loop.models import (
    LoopLimits,
    RunInput,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolBatchOutcome,
)
from .observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely


@dataclass(frozen=True, slots=True)
class CapabilityFailure:
    """One bounded domain-owned failure ready for common result rendering."""

    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("capability failure code must be non-empty text")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("capability failure message must be non-empty text")
        object.__setattr__(
            self,
            "details",
            FrozenJsonObject.from_mapping(self.details),
        )


@dataclass(frozen=True, slots=True)
class SideEffectPlan:
    """Domain-owned approval and recheck semantics for one exact preflight."""

    approval_required: bool = True
    approval_arguments: FrozenJsonObject | None = None
    approval_reason: str = "Allow this exact side-effecting tool invocation once?"
    recheck_after_approval: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.approval_required, bool):
            raise TypeError("approval_required must be a boolean")
        if self.approval_arguments is not None and not isinstance(
            self.approval_arguments, FrozenJsonObject
        ):
            raise TypeError("approval_arguments must be FrozenJsonObject or None")
        if not isinstance(self.approval_reason, str) or not self.approval_reason:
            raise ValueError("approval_reason must be non-empty text")
        if not isinstance(self.recheck_after_approval, bool):
            raise TypeError("recheck_after_approval must be a boolean")


class CapabilityDomain(Protocol):
    """Narrow static owner contract exercised by current native families."""

    @property
    def domain_owner_id(self) -> str: ...

    @property
    def declarations(self) -> CapabilityDeclarations: ...

    async def project(self, run: RunInput) -> tuple[str, ...]: ...

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]: ...

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
    ) -> FrozenJsonObject: ...

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan: ...

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
    ) -> ToolOutput: ...

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None: ...


@dataclass(frozen=True, slots=True)
class _ProjectedTool:
    view: ToolView
    capability: Capability
    domain_owner_id: str


class _ToolExecutionInterrupted(Exception):
    def __init__(
        self,
        result: ToolResultBlock,
        kind: ToolBatchInterruption,
        certainty: ToolBatchCertainty,
    ) -> None:
        super().__init__(kind.value)
        self.result = result
        self.kind = kind
        self.certainty = certainty


class _ToolOutcomeUnknown(RuntimeError):
    pass


class CapabilityRuntime:
    """Apply common execution mechanics to statically composed domains."""

    def __init__(
        self,
        registry: CapabilityRegistry,
        domains: tuple[CapabilityDomain, ...],
        *,
        approval_handler: ApprovalHandler | None = None,
        mutation_lock: asyncio.Lock | None = None,
        observer: AgentObserver | None = None,
        clock: Callable[[], datetime] | None = None,
        artifacts: AgentHomeArtifactStore | None = None,
        limits: LoopLimits = LoopLimits(),
        side_effect_recovery_timeout_seconds: float | None = None,
    ) -> None:
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("registry must be CapabilityRegistry")
        domains = tuple(domains)
        owners: dict[str, CapabilityDomain] = {}
        for domain in domains:
            owner_id = domain.domain_owner_id
            if not isinstance(owner_id, str) or not owner_id:
                raise ValueError("domain_owner_id must be non-empty text")
            if owner_id in owners:
                raise ValueError(f"duplicate capability domain: {owner_id}")
            if domain.declarations.domain_owner_id != owner_id:
                raise ValueError(f"domain declaration owner differs: {owner_id}")
            registry.validate_declarations(domain.declarations)
            owners[owner_id] = domain
        if set(owners) != registry.domain_owner_ids:
            raise ValueError(
                "runtime domains must exactly match registry domain owners"
            )
        if approval_handler is not None and not callable(approval_handler):
            raise TypeError("approval_handler must be callable or None")
        if mutation_lock is not None and not isinstance(mutation_lock, asyncio.Lock):
            raise TypeError("mutation_lock must be an asyncio.Lock or None")
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable or None")
        if not isinstance(limits, LoopLimits):
            raise TypeError("limits must be LoopLimits")
        recovery_timeout = (
            limits.side_effect_recovery_timeout_seconds
            if side_effect_recovery_timeout_seconds is None
            else side_effect_recovery_timeout_seconds
        )
        if (
            not isinstance(recovery_timeout, (int, float))
            or isinstance(recovery_timeout, bool)
            or not 0 < float(recovery_timeout) <= 60
        ):
            raise ValueError(
                "side_effect_recovery_timeout_seconds must be positive and at most 60"
            )
        self._registry = registry
        self._domains = owners
        self._approval_handler = approval_handler
        self._mutation_lock = mutation_lock or asyncio.Lock()
        self._observer = observer
        self._clock = clock or (lambda: datetime.now(UTC))
        self._artifacts = artifacts
        self._limits = limits
        self._side_effect_recovery_timeout_seconds = float(recovery_timeout)

    async def definitions(self, run: RunInput) -> tuple[ToolDefinition, ...]:
        projected = await self._projection(run)
        return tuple(self._registry.tool_definition(name) for name in sorted(projected))

    async def execute_all(
        self,
        run: RunInput,
        calls: tuple[ToolCall, ...],
    ) -> ToolBatchOutcome:
        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        calls = tuple(calls)
        if any(not isinstance(call, ToolCall) for call in calls):
            raise TypeError("calls must contain ToolCall records")
        if len(calls) > self._limits.max_tool_calls_per_response:
            raise ValueError("tool batch exceeds max_tool_calls_per_response")
        projected = await self._projection(run)
        results: list[ToolResultBlock | None] = [None] * len(calls)
        started = [False] * len(calls)
        reads: list[tuple[int, ToolCall]] = []
        read_gate = asyncio.Semaphore(self._limits.max_parallel_reads)
        source_gates: dict[str, asyncio.Semaphore] = {}

        async def execute_read(index: int, call: ToolCall) -> ToolResultBlock:
            source_key = _source_pressure_key(run, call)
            source_gate = source_gates.setdefault(
                source_key,
                asyncio.Semaphore(self._limits.max_parallel_reads_per_source),
            )
            async with read_gate:
                async with source_gate:
                    started[index] = True
                    return await self._execute_one(run, call, projected)

        async def finish_reads() -> ToolBatchInterruption | None:
            if not reads:
                return None
            indexes = tuple(index for index, _ in reads)
            read_calls = tuple(call for _, call in reads)
            tasks = tuple(
                asyncio.create_task(execute_read(index, call)) for index, call in reads
            )
            try:
                await asyncio.wait(tasks)
            except asyncio.CancelledError as error:
                interruption = _cancel_interruption(error)
                for task in tasks:
                    if not task.done():
                        task.cancel(interruption.value)
                settled, pending = await _settle_cancelled_reads(
                    tasks,
                    timeout_seconds=self._side_effect_recovery_timeout_seconds,
                )
                for task in pending:
                    task.add_done_callback(_consume_read_task)
                for index, call, task in zip(
                    indexes,
                    read_calls,
                    tasks,
                    strict=True,
                ):
                    value: ToolResultBlock | None = None
                    if task in settled and not task.cancelled():
                        try:
                            value = task.result()
                        except BaseException:
                            pass
                    results[index] = value or _interruption_result(
                        call,
                        interruption,
                        started=started[index],
                        outcome_unknown=False,
                    )
                reads.clear()
                return interruption
            for index, task in zip(indexes, tasks, strict=True):
                results[index] = task.result()
            reads.clear()
            return None

        for index, call in enumerate(calls):
            if self._is_side_effecting(call, projected):
                read_interruption = await finish_reads()
                if read_interruption is not None:
                    return _interrupted_batch(
                        calls,
                        results,
                        started,
                        read_interruption,
                        ToolBatchCertainty.DEFINITE,
                    )
                started[index] = True
                try:
                    results[index] = await self._execute_one(run, call, projected)
                except _ToolExecutionInterrupted as interrupted:
                    results[index] = interrupted.result
                    return _interrupted_batch(
                        calls,
                        results,
                        started,
                        interrupted.kind,
                        interrupted.certainty,
                    )
                except asyncio.CancelledError as error:
                    interruption = _cancel_interruption(error)
                    results[index] = _interruption_result(
                        call,
                        interruption,
                        started=True,
                        outcome_unknown=False,
                    )
                    return _interrupted_batch(
                        calls,
                        results,
                        started,
                        interruption,
                        ToolBatchCertainty.DEFINITE,
                    )
            else:
                reads.append((index, call))
        read_interruption = await finish_reads()
        if read_interruption is not None:
            return _interrupted_batch(
                calls,
                results,
                started,
                read_interruption,
                ToolBatchCertainty.DEFINITE,
            )
        if any(result is None for result in results):
            raise RuntimeError("tool result scheduling left an incomplete call")
        return ToolBatchOutcome(cast(tuple[ToolResultBlock, ...], tuple(results)))

    async def _projection(self, run: RunInput) -> dict[str, _ProjectedTool]:
        projected: dict[str, _ProjectedTool] = {}
        for owner_id in sorted(self._domains):
            domain = self._domains[owner_id]
            for name in await domain.project(run):
                if name in projected:
                    raise ValueError(f"tool projected by multiple domains: {name}")
                view, capability, resolved_owner = self._registry.resolve_tool_owner(
                    name
                )
                if resolved_owner != owner_id:
                    raise ValueError(f"tool projected by the wrong domain: {name}")
                projected[name] = _ProjectedTool(view, capability, resolved_owner)
        return projected

    async def _execute_one(
        self,
        run: RunInput,
        call: ToolCall,
        projected: Mapping[str, _ProjectedTool],
    ) -> ToolResultBlock:
        started = (
            asyncio.get_running_loop().time() if self._observer is not None else None
        )
        projection = projected.get(call.name)
        capability = None if projection is None else projection.capability
        self._emit_tool_started(run, call, capability)
        interruption_kind: ToolBatchInterruption | None = None
        outcome_certainty = ToolBatchCertainty.DEFINITE
        domain: CapabilityDomain | None = None
        try:
            if projection is None:
                result = _error(
                    call,
                    "tool_not_available",
                    "The requested tool is not available for the current execution scope.",
                    {"tool_name": call.name},
                )
                self._emit_tool_completed(run, call, result, started)
                return result
            view, resolved, owner_id = self._registry.resolve_tool_owner(call.name)
            if (
                view != projection.view
                or resolved != projection.capability
                or owner_id != projection.domain_owner_id
            ):
                raise ValueError("tool projection identity changed")
            capability = resolved
            domain = self._domains[owner_id]
            raw_arguments = domain.normalize_arguments(capability, call.arguments)
            arguments = self._registry.validate_arguments(
                capability.id,
                raw_arguments,
            )
            arguments = await domain.prepare_call(
                run,
                call,
                capability,
                arguments,
            )
            resolved_capability, executor = self._registry.resolve_execution(
                capability.id
            )
            if (
                resolved_capability != capability
                or view.capability_id != capability.id
                or self._registry.resolve_domain_owner(capability.id) != owner_id
            ):
                raise ValueError("tool execution identity changed")
            execution = ToolExecution(
                run_id=run.id,
                call_id=call.id,
                capability_id=capability.id,
                arguments=arguments,
                conversation_id=run.conversation_id or run.id,
            )
            if capability.side_effecting:
                (
                    result,
                    interruption_kind,
                    outcome_certainty,
                ) = await self._execute_side_effect(
                    run,
                    call,
                    capability,
                    executor,
                    execution,
                    arguments,
                    domain,
                )
            else:
                candidate = await executor.execute(execution)
                if not isinstance(candidate, ToolOutput):
                    raise ToolOutputValidationError(
                        "executor did not return ToolOutput"
                    )
                output = candidate
                output = await domain.finalize_output(
                    run,
                    call,
                    capability,
                    arguments,
                    output,
                )
                output = self._registry.validate_output(capability.id, output)
                artifact_ref = await self._commit_artifact_output(
                    run,
                    call,
                    capability,
                    output,
                )
                result = _classified_success(call, output, artifact_ref=artifact_ref)
        except _ToolExecutionInterrupted:
            raise
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            result = self._exception_result(call, error, domain)
        if not result.is_error:
            bound_issue = _tool_result_bound_issue(result, self._limits)
            if bound_issue is not None:
                code, message, details = bound_issue
                result = _error(call, code, message, details)
        self._emit_tool_completed(run, call, result, started)
        if interruption_kind is not None:
            raise _ToolExecutionInterrupted(
                result,
                interruption_kind,
                outcome_certainty,
            )
        return result

    async def _execute_side_effect(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        executor: Executor,
        execution: ToolExecution,
        arguments: FrozenJsonObject,
        domain: CapabilityDomain,
    ) -> tuple[
        ToolResultBlock,
        ToolBatchInterruption | None,
        ToolBatchCertainty,
    ]:
        if (
            capability.access_mode is not AccessMode.WRITE
            or not capability.side_effecting
        ):
            raise ValueError("side-effect execution requires a write capability")
        preflight = getattr(executor, "preflight", None)
        if not callable(preflight):
            raise ValueError("side-effecting executor must provide preflight")
        side_effect = cast(SideEffectExecutor, executor)
        fingerprint = await side_effect.preflight(execution)
        if not isinstance(fingerprint, FrozenJsonObject):
            raise ValueError("side-effect preflight must return FrozenJsonObject")
        plan = await domain.side_effect_plan(
            run,
            call,
            capability,
            execution,
            fingerprint,
        )
        if plan.approval_required:
            if self._approval_handler is None:
                return (
                    _error(
                        call,
                        "approval_required",
                        "This side effect requires an approval handler.",
                        {"capability_id": capability.id},
                    ),
                    None,
                    ToolBatchCertainty.DEFINITE,
                )
            request = ApprovalRequest(
                run_id=run.id,
                call_id=call.id,
                tool_name=call.name,
                capability_id=capability.id,
                arguments=plan.approval_arguments or arguments,
                reason=plan.approval_reason,
            )
            self._emit_approval_requested(run, call, capability)
            try:
                decision = await self._approval_handler(request)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._emit_approval_decided(run, call, "failed")
                return (
                    _error(
                        call,
                        "approval_failed",
                        "The approval handler failed closed.",
                        {"capability_id": capability.id},
                    ),
                    None,
                    ToolBatchCertainty.DEFINITE,
                )
            if not isinstance(decision, ApprovalDecision):
                self._emit_approval_decided(run, call, "failed")
                return (
                    _error(
                        call,
                        "approval_failed",
                        "The approval handler returned an invalid decision.",
                        {"capability_id": capability.id},
                    ),
                    None,
                    ToolBatchCertainty.DEFINITE,
                )
            if decision is ApprovalDecision.DENY:
                self._emit_approval_decided(run, call, "denied")
                return (
                    _error(
                        call,
                        "approval_denied",
                        "The side effect was denied.",
                        {"capability_id": capability.id},
                    ),
                    None,
                    ToolBatchCertainty.DEFINITE,
                )
            self._emit_approval_decided(run, call, "approved")
        return await self._execute_preflighted_side_effect(
            run,
            call,
            capability,
            side_effect,
            execution,
            arguments,
            fingerprint,
            plan,
            domain,
        )

    async def _execute_preflighted_side_effect(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        side_effect: SideEffectExecutor,
        execution: ToolExecution,
        arguments: FrozenJsonObject,
        fingerprint: FrozenJsonObject,
        plan: SideEffectPlan,
        domain: CapabilityDomain,
    ) -> tuple[
        ToolResultBlock,
        ToolBatchInterruption | None,
        ToolBatchCertainty,
    ]:
        async with self._mutation_lock:
            if plan.recheck_after_approval:
                try:
                    current = await side_effect.preflight(execution)
                    if not isinstance(current, FrozenJsonObject):
                        raise ValueError(
                            "side-effect preflight must return FrozenJsonObject"
                        )
                    await domain.side_effect_plan(
                        run,
                        call,
                        capability,
                        execution,
                        current,
                    )
                except asyncio.CancelledError:
                    raise
                except BaseException as error:
                    if domain.normalize_error(call, error) is None and not isinstance(
                        error, (CapabilityInputError, DaitaError, ArtifactError)
                    ):
                        raise
                    return (
                        _error(
                            call,
                            "state_changed",
                            "The validated state changed while approval was pending.",
                            {"capability_id": capability.id},
                        ),
                        None,
                        ToolBatchCertainty.DEFINITE,
                    )
                if current != fingerprint:
                    return (
                        _error(
                            call,
                            "state_changed",
                            "The validated state changed while approval was pending.",
                            {"capability_id": capability.id},
                        ),
                        None,
                        ToolBatchCertainty.DEFINITE,
                    )
            candidate, execution_error, interruption_kind, outcome_certainty = (
                await _execute_definitely(
                    side_effect,
                    execution,
                    recovery_timeout_seconds=(
                        self._side_effect_recovery_timeout_seconds
                    ),
                )
            )
            if execution_error is not None:
                return (
                    self._exception_result(call, execution_error, domain),
                    interruption_kind,
                    outcome_certainty,
                )
            if not isinstance(candidate, ToolOutput):
                raise ToolOutputValidationError("executor did not return ToolOutput")
            output = candidate
            output = await domain.finalize_output(
                run,
                call,
                capability,
                arguments,
                output,
            )
            output = self._registry.validate_output(capability.id, output)
            if output.artifact is not None or capability.artifact_policy is not None:
                raise ToolOutputValidationError(
                    "side-effect capability cannot produce an artifact draft"
                )
            return (
                _classified_success(call, output),
                interruption_kind,
                outcome_certainty,
            )

    async def _commit_artifact_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        output: ToolOutput,
    ) -> ArtifactRef | None:
        policy = capability.artifact_policy
        draft = output.artifact
        if policy is None:
            if draft is not None:
                raise ToolOutputValidationError(
                    "capability without artifact policy returned a draft"
                )
            return None
        if draft is None:
            if policy.artifact_required:
                raise ToolOutputValidationError(
                    "artifact-producing capability omitted its required draft"
                )
            return None
        if policy.max_artifact_count != 1:
            raise ToolOutputValidationError(
                "artifact draft exceeds the capability artifact count"
            )
        if draft.media_type not in policy.allowed_media_types:
            raise ToolOutputValidationError(
                "artifact draft media type is outside the capability policy"
            )
        if (
            len(draft.content) > policy.max_bytes_per_artifact
            or len(draft.content) > policy.max_total_bytes_per_call
        ):
            raise ToolOutputValidationError(
                "artifact draft bytes exceed the capability policy"
            )
        try:
            canonical_artifact_filename(
                draft.suggested_filename,
                draft.media_type,
                policy.allowed_extensions,
            )
        except ArtifactError as error:
            raise ToolOutputValidationError(
                "artifact draft filename or extension violates the capability policy"
            ) from error
        if self._artifacts is None:
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact storage is unavailable.",
                {"stage": "composition"},
            )
        return await self._artifacts.commit(
            draft,
            policy,
            run_id=run.id,
            conversation_id=run.conversation_id or run.id,
            call_id=call.id,
            capability_id=capability.id,
        )

    def _exception_result(
        self,
        call: ToolCall,
        error: BaseException,
        domain: CapabilityDomain | None,
    ) -> ToolResultBlock:
        if domain is not None:
            normalized = domain.normalize_error(call, error)
            if normalized is not None:
                return _error(
                    call,
                    normalized.code,
                    normalized.message,
                    normalized.details,
                )
        if isinstance(error, _ToolOutcomeUnknown):
            return _error(
                call,
                "outcome_unknown",
                "The tool action started, but its authoritative outcome was not "
                "available within the bounded recovery wait.",
                {
                    "execution_state": "started",
                    "outcome_certainty": ToolBatchCertainty.OUTCOME_UNKNOWN.value,
                },
            )
        if isinstance(error, CapabilityInputError):
            return _error(call, error.code, str(error), error.details)
        if isinstance(error, ToolOutputValidationError):
            return _error(call, "invalid_tool_result", str(error))
        if isinstance(error, ArtifactError):
            return _error(call, error.code, error.message, error.details)
        if isinstance(error, DaitaError):
            details = getattr(error, "details", None)
            return _error(
                call,
                error.error_code,
                str(error),
                details if isinstance(details, Mapping) else None,
            )
        return _error(
            call,
            "tool_execution_failed",
            "The tool could not complete because of an unexpected internal error.",
        )

    def _is_side_effecting(
        self,
        call: ToolCall,
        projected: Mapping[str, _ProjectedTool],
    ) -> bool:
        projection = projected.get(call.name)
        return projection is not None and projection.capability.side_effecting

    def _emit_tool_started(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability | None,
    ) -> None:
        data: dict[str, object] = {"call_id": call.id, "tool_name": call.name}
        if capability is not None:
            data["capability_id"] = capability.id
        self._emit(AgentEventKind.TOOL_STARTED, run, data)

    def _emit_approval_requested(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
    ) -> None:
        self._emit(
            AgentEventKind.APPROVAL_REQUESTED,
            run,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "capability_id": capability.id,
            },
        )

    def _emit_approval_decided(
        self,
        run: RunInput,
        call: ToolCall,
        outcome: str,
    ) -> None:
        self._emit(
            AgentEventKind.APPROVAL_DECIDED,
            run,
            {"call_id": call.id, "outcome": outcome},
        )

    def _emit_tool_completed(
        self,
        run: RunInput,
        call: ToolCall,
        result: ToolResultBlock,
        started: float | None,
    ) -> None:
        if self._observer is None:
            return
        assert started is not None
        self._emit(
            AgentEventKind.TOOL_COMPLETED,
            run,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "duration_ms": _duration_ms(started),
                "success": not result.is_error,
                "error_code": _result_error_code(result),
            },
        )

    def _emit(
        self,
        kind: AgentEventKind,
        run: RunInput,
        data: Mapping[str, object],
    ) -> None:
        if self._observer is None:
            return
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
        _emit_safely(self._observer, event)


async def _execute_definitely(
    executor: SideEffectExecutor,
    execution: ToolExecution,
    *,
    recovery_timeout_seconds: float,
) -> tuple[
    ToolOutput | None,
    BaseException | None,
    ToolBatchInterruption | None,
    ToolBatchCertainty,
]:
    worker = asyncio.create_task(executor.execute(execution))
    interruption: ToolBatchInterruption | None = None
    recovery_deadline: float | None = None
    while not worker.done():
        if recovery_deadline is not None:
            remaining = recovery_deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                worker.cancel(interruption.value if interruption is not None else None)
                worker.add_done_callback(_consume_background_task)
                return (
                    None,
                    _ToolOutcomeUnknown(),
                    interruption,
                    ToolBatchCertainty.OUTCOME_UNKNOWN,
                )
            done, _pending = await asyncio.wait((worker,), timeout=remaining)
            if not done:
                worker.cancel(interruption.value if interruption is not None else None)
                worker.add_done_callback(_consume_background_task)
                return (
                    None,
                    _ToolOutcomeUnknown(),
                    interruption,
                    ToolBatchCertainty.OUTCOME_UNKNOWN,
                )
            break
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError as error:
            interruption = _cancel_interruption(error)
            recovery_deadline = (
                asyncio.get_running_loop().time() + recovery_timeout_seconds
            )
            continue
        except BaseException:
            if worker.done():
                break
            raise
    try:
        return worker.result(), None, interruption, ToolBatchCertainty.DEFINITE
    except BaseException as error:
        return None, error, interruption, ToolBatchCertainty.DEFINITE


def _consume_background_task(task: asyncio.Task[ToolOutput]) -> None:
    try:
        task.exception()
    except BaseException:
        pass


async def _settle_cancelled_reads(
    tasks: tuple[asyncio.Task[ToolResultBlock], ...],
    *,
    timeout_seconds: float,
) -> tuple[
    set[asyncio.Task[ToolResultBlock]],
    set[asyncio.Task[ToolResultBlock]],
]:
    waiter = asyncio.create_task(asyncio.wait(tasks, timeout=timeout_seconds))
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            continue
    done, pending = waiter.result()
    return set(done), set(pending)


def _consume_read_task(task: asyncio.Task[ToolResultBlock]) -> None:
    try:
        task.exception()
    except BaseException:
        pass


def _cancel_interruption(error: asyncio.CancelledError) -> ToolBatchInterruption:
    return (
        ToolBatchInterruption.DEADLINE
        if error.args and error.args[0] == ToolBatchInterruption.DEADLINE.value
        else ToolBatchInterruption.CANCELLED
    )


def _interruption_result(
    call: ToolCall,
    interruption: ToolBatchInterruption,
    *,
    started: bool,
    outcome_unknown: bool,
) -> ToolResultBlock:
    if outcome_unknown:
        return _error(
            call,
            "outcome_unknown",
            "The tool action started, but its authoritative outcome was not "
            "available within the bounded recovery wait.",
            {
                "interruption_kind": interruption.value,
                "execution_state": "started",
                "outcome_certainty": ToolBatchCertainty.OUTCOME_UNKNOWN.value,
            },
        )
    if started:
        return _error(
            call,
            "tool_call_interrupted",
            "The started read was interrupted before it returned a result.",
            {
                "interruption_kind": interruption.value,
                "execution_state": "started",
                "outcome_certainty": ToolBatchCertainty.DEFINITE.value,
            },
        )
    return _error(
        call,
        "tool_call_not_started",
        "The tool call did not start before its batch was interrupted.",
        {
            "interruption_kind": interruption.value,
            "execution_state": "not_started",
            "outcome_certainty": ToolBatchCertainty.DEFINITE.value,
        },
    )


def _interrupted_batch(
    calls: tuple[ToolCall, ...],
    results: list[ToolResultBlock | None],
    started: list[bool],
    interruption: ToolBatchInterruption,
    certainty: ToolBatchCertainty,
) -> ToolBatchOutcome:
    ordered = tuple(
        (
            result
            if result is not None
            else _interruption_result(
                call,
                interruption,
                started=started[index],
                outcome_unknown=False,
            )
        )
        for index, (call, result) in enumerate(zip(calls, results, strict=True))
    )
    return ToolBatchOutcome(
        ordered_results=ordered,
        interruption_kind=interruption,
        outcome_certainty=certainty,
    )


def _result_error_code(result: ToolResultBlock) -> str | None:
    if not result.is_error:
        return None
    error = result.output.get("error")
    if not isinstance(error, Mapping):
        return "unknown_tool_error"
    code = error.get("code")
    return code if isinstance(code, str) else "unknown_tool_error"


def _duration_ms(started: float) -> int:
    elapsed = asyncio.get_running_loop().time() - started
    return max(0, int(elapsed * 1_000))


def _source_pressure_key(run: RunInput, call: ToolCall) -> str:
    source_id = call.arguments.get("source_id")
    if isinstance(source_id, str):
        return source_id
    if run.source_id is not None:
        return run.source_id
    return "__agent_local__"


def _tool_result_bound_issue(
    result: ToolResultBlock,
    limits: LoopLimits,
) -> tuple[str, str, Mapping[str, object]] | None:
    result_bytes = len(canonical_json(result.output).encode("utf-8"))
    if result_bytes > limits.max_tool_result_bytes:
        return (
            "tool_result_too_large",
            "The tool result exceeded the fixed model-visible byte bound.",
            {
                "maximum_bytes": limits.max_tool_result_bytes,
                "observed_bytes": result_bytes,
            },
        )
    result_depth = _json_depth(result.output)
    if result_depth > limits.max_tool_result_depth:
        return (
            "tool_result_too_deep",
            "The tool result exceeded the fixed model-visible nesting bound.",
            {
                "maximum_depth": limits.max_tool_result_depth,
                "observed_depth": result_depth,
            },
        )
    return None


def _json_depth(value: object) -> int:
    if isinstance(value, Mapping):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, (tuple, list)):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _classified_success(
    call: ToolCall,
    output: ToolOutput,
    *,
    artifact_ref: ArtifactRef | None = None,
) -> ToolResultBlock:
    if output.sensitivity is None or not output.sensitivity_provenance:
        return _error(
            call,
            "result_classification_unavailable",
            "The tool result could not be classified by its owning domain.",
            {"capability_id": output.kind},
        )
    result: dict[str, object] = {"kind": output.kind, "data": output.data}
    if artifact_ref is not None:
        result["artifact"] = artifact_ref_to_mapping(artifact_ref)
        result["delivery_status"] = "not_delivered"
    return ToolResultBlock(
        call_id=call.id,
        output=result,
        sensitivity=output.sensitivity,
        sensitivity_provenance=output.sensitivity_provenance,
    )


def _error(
    call: ToolCall,
    code: str,
    message: str,
    details: Mapping[str, object] | None = None,
) -> ToolResultBlock:
    return ToolResultBlock(
        call_id=call.id,
        is_error=True,
        output={
            "error": {
                "code": code,
                "message": message,
                "details": {} if details is None else details,
            }
        },
    )


__all__ = [
    "CapabilityDomain",
    "CapabilityFailure",
    "CapabilityRuntime",
    "SideEffectPlan",
]
