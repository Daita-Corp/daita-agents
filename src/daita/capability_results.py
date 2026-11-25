"""Tool-result shaping, interruption and scope checks for the runtime."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import replace
from hashlib import sha256
from typing import TYPE_CHECKING, TypeVar

from ._json import canonical_json
from .artifacts.models import ArtifactRef, artifact_ref_to_mapping
from .capabilities import (
    Capability,
    CapabilityInputError,
    RecoveryKind,
    ToolOutput,
    ToolRecovery,
    recovery_details,
)
from .llm.models import (
    CanonicalMessage,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
    model_tool_result,
    tool_result_refs,
)
from .loop.models import (
    LoopLimits,
    RunInput,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolBatchOutcome,
)

if TYPE_CHECKING:
    from .capability_runtime import _ResolvedCall
    from .storage.sqlite_records import EffectReceipt


# The result_ref each call of the executing batch receives in the transcript, so
# TOOL_COMPLETED can report the exact size of the model-facing rendering.
_BATCH_RESULT_REFS: ContextVar[Mapping[str, str] | None] = ContextVar(
    "daita_batch_result_refs", default=None
)


def _batch_result_refs(
    messages: tuple[CanonicalMessage, ...], calls: tuple[ToolCall, ...]
) -> dict[str, str]:
    """Each call's result_ref once its result is appended in call order."""

    if any(not isinstance(call, ToolCall) for call in calls):
        return {}
    prior = len(tool_result_refs(tuple(messages)).by_call_id)
    return {call.id: f"r{prior + index}" for index, call in enumerate(calls, 1)}


def _result_bytes(result: ToolResultBlock) -> dict[str, int | None]:
    """Canonical stored-output size and model-facing rendering size.

    Code-owned internal calls are never rendered to a model, so their model
    size is null rather than an invented number.
    """

    refs = _BATCH_RESULT_REFS.get()
    stored = len(canonical_json(result.output).encode("utf-8"))
    if refs is None:
        return {"stored": stored, "model": None}
    rendered = model_tool_result(result, refs.get(result.call_id))
    return {"stored": stored, "model": len(canonical_json(rendered).encode("utf-8"))}


def _control_success(
    call: ToolCall,
    kind: str,
    data: Mapping[str, object],
    *,
    sensitivity: ModelSensitivity,
    run_id: str,
    catalog_digest: str,
) -> ToolResultBlock:
    return ToolResultBlock(
        call_id=call.id,
        output={"kind": kind, "data": data},
        sensitivity=sensitivity,
        sensitivity_provenance={
            "authority": "tool_catalog_control",
            "run_id": run_id,
            "catalog_digest": catalog_digest,
            "control_name": call.name,
        },
    )


def _consume_background_task(task: asyncio.Task[ToolOutput]) -> None:
    try:
        task.exception()
    except BaseException:
        pass


_ReadResultT = TypeVar("_ReadResultT")


async def _settle_cancelled_reads(
    tasks: tuple[asyncio.Task[_ReadResultT], ...],
    *,
    timeout_seconds: float,
) -> tuple[
    set[asyncio.Task[_ReadResultT]],
    set[asyncio.Task[_ReadResultT]],
]:
    waiter = asyncio.create_task(asyncio.wait(tasks, timeout=timeout_seconds))
    while not waiter.done():
        try:
            await asyncio.shield(waiter)
        except asyncio.CancelledError:
            continue
    done, pending = waiter.result()
    return set(done), set(pending)


def _consume_read_task(task: asyncio.Task[object]) -> None:
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
    resolved_calls: tuple[_ResolvedCall, ...],
    results: list[ToolResultBlock | None],
    started: list[bool],
    interruption: ToolBatchInterruption,
    certainty: ToolBatchCertainty,
    limits: LoopLimits,
) -> ToolBatchOutcome:
    ordered = tuple(
        (
            result
            if result is not None
            else _interruption_result(
                resolved.outer_call,
                interruption,
                started=started[index],
                outcome_unknown=False,
            )
        )
        for index, (resolved, result) in enumerate(
            zip(resolved_calls, results, strict=True)
        )
    )
    bounded = tuple(
        _bounded_tool_result(
            resolved.outer_call,
            result,
            limits,
        )
        for resolved, result in zip(resolved_calls, ordered, strict=True)
    )
    return ToolBatchOutcome(
        ordered_results=bounded,
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


def _validate_run_execution_scope(
    run: RunInput,
    capability: Capability,
    sensitivity: ModelSensitivity,
) -> None:
    scope = run.execution_scope
    if scope is None:
        return
    source_identity_allowed = set(run.source_scope_ids) <= set(scope.allowed_source_ids)
    if (
        scope.agent_id != run.agent_id
        or not source_identity_allowed
        or not scope.allows(capability)
    ):
        raise CapabilityInputError(
            "execution_scope_violation",
            "The requested capability is outside this run's immutable execution scope.",
            {"capability_id": capability.id},
        )
    if sensitivity.routing_rank > scope.sensitivity_ceiling.routing_rank:
        raise CapabilityInputError(
            "execution_scope_sensitivity_exceeded",
            "The current request sensitivity exceeds this run's immutable ceiling.",
            {"capability_id": capability.id},
        )


def _validate_output_execution_scope(
    run: RunInput,
    capability: Capability,
    output: ToolOutput,
) -> None:
    scope = run.execution_scope
    if scope is None or output.sensitivity is None:
        return
    if output.sensitivity.routing_rank > scope.sensitivity_ceiling.routing_rank:
        raise CapabilityInputError(
            "execution_scope_sensitivity_exceeded",
            "The validated result exceeds this run's immutable sensitivity ceiling.",
            {"capability_id": capability.id},
        )


def _source_pressure_key(run: RunInput, call: ToolCall) -> str:
    source = call.arguments.get("source", call.arguments.get("source_id"))
    if isinstance(source, str):
        # A source's display name and its full ID share one per-source gate.
        scope = run.resolved_source_scope
        named = () if scope is None else scope.sources_named(source)
        return named[0] if len(named) == 1 else source
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


def _with_effect_reference(
    result: ToolResultBlock, receipt: EffectReceipt
) -> ToolResultBlock:
    return replace(
        result,
        output={
            **result.output,
            "effect_receipt": {
                "receipt_id": receipt.receipt_id,
                "receipt_digest": receipt.receipt_digest,
                "outcome": receipt.outcome.value,
                "evidence_basis": receipt.evidence_basis.value,
            },
        },
    )


def _effect_duplicate_result(call: ToolCall, receipt: EffectReceipt) -> ToolResultBlock:
    return _with_effect_reference(
        _error(
            call,
            "effect_already_reserved",
            "This operation was already reserved and was not dispatched again.",
            recovery_details(
                ToolRecovery(
                    RecoveryKind.TERMINAL,
                    "This call already has a receipt; report its recorded outcome.",
                ),
                # The earlier reservation owns the outcome; this attempt proves nothing.
                {"execution_disposition": "unknown"},
            ),
        ),
        receipt,
    )


def _bounded_tool_result(
    call: ToolCall,
    result: ToolResultBlock,
    limits: LoopLimits,
) -> ToolResultBlock:
    issue = _tool_result_bound_issue(result, limits)
    if issue is None:
        return result
    code, message, details = issue
    bounded = _error(call, code, message, details)
    if _tool_result_bound_issue(bounded, limits) is None:
        return bounded
    fallback = _error(call, code, "The tool result exceeded its fixed bound.")
    if _tool_result_bound_issue(fallback, limits) is not None:
        raise ValueError("loop limits cannot represent the bounded tool error")
    return fallback


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
    artifact_refs: tuple[ArtifactRef, ...] = (),
) -> ToolResultBlock:
    if output.sensitivity is None or not output.sensitivity_provenance:
        return _error(
            call,
            "result_classification_unavailable",
            "The tool result could not be classified by its owning domain.",
            {"capability_id": output.kind},
        )
    result: dict[str, object] = {"kind": output.kind, "data": output.data}
    if artifact_refs:
        result["artifact"] = artifact_ref_to_mapping(artifact_refs[0])
        if len(artifact_refs) > 1:
            result["artifacts"] = [
                artifact_ref_to_mapping(ref) for ref in artifact_refs
            ]
        result["delivery_status"] = "not_delivered"
    return ToolResultBlock(
        call_id=call.id,
        output=result,
        sensitivity=output.sensitivity,
        sensitivity_provenance=output.sensitivity_provenance,
    )


def _with_execution_lineage(
    result: ToolResultBlock,
    capability: Capability,
) -> ToolResultBlock:
    """Persist stable code-owned capability lineage with every executed result."""

    return replace(
        result,
        capability_id=capability.id,
        executor_id=capability.executor_id,
        output_sha256=(
            None
            if result.is_error
            else "sha256:"
            + sha256(canonical_json(result.output).encode("utf-8")).hexdigest()
        ),
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


def _with_execution_disposition(
    result: ToolResultBlock,
    disposition: str,
) -> ToolResultBlock:
    """Record a proven dispatch fact without replacing one the owner set."""

    error = result.output.get("error")
    if not isinstance(error, Mapping):
        return result
    details = error.get("details")
    current = dict(details) if isinstance(details, Mapping) else {}
    if "execution_disposition" in current:
        return result
    current["execution_disposition"] = disposition
    return replace(
        result,
        output={**result.output, "error": {**error, "details": current}},
    )
