"""Run analysis-session child reads through the single-call runtime boundary."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from typing import TYPE_CHECKING, cast

from ._json import canonical_json
from .artifacts.models import (
    ArtifactRef,
    artifact_ref_from_mapping,
    artifact_ref_to_mapping,
)
from .capabilities import (
    tool_call_mode,
    tool_route_arguments,
    tool_route_unknown_arguments,
)
from .capability_domain_calls import _domain_project
from .capability_results import (
    _consume_read_task,
    _error,
    _settle_cancelled_reads,
    _source_pressure_key,
)
from .llm.models import ModelSensitivity, ToolCall, ToolResultBlock
from .loop.analysis import (
    AnalysisEvidence,
    AnalysisTraceCapacityError,
    ProgrammaticReadBroker,
    ProgrammaticReadResult,
)
from .loop.models import RunInput
from .loop.session import RunSession

if TYPE_CHECKING:
    from .capability_runtime import CapabilityRuntime, StepToolProjection
    from .domains.analysis.domain import AnalysisCapabilityDomain


def programmatic_reader(
    runtime: CapabilityRuntime,
    session: RunSession,
    run: RunInput,
    parent: ToolCall,
    projection: StepToolProjection,
):
    """Freeze one parent surface and share the ordinary single-call boundary."""
    entries = {
        entry.view.name: entry
        for entry in projection.callable_entries
        if runtime._analysis_read_eligible(entry.domain_owner_id, entry.capability)
        or runtime._analysis_capture_eligible(entry.domain_owner_id, entry.capability)
    }
    # A routed child executes its selected capability with that route's
    # arguments; the child call itself keeps the arguments it was given.
    targets: dict[str, ToolCall] = {}
    sequence = 0
    child_limit = cast(
        "AnalysisCapabilityDomain", runtime._domains["analysis"]
    ).child_call_limit

    def allocate(
        name: str, arguments: Mapping[str, object], sensitivity: ModelSensitivity
    ):
        nonlocal sequence
        sequence += 1
        child = ToolCall(
            id=f"analysis-child-{secrets.token_hex(16)}",
            name=name,
            arguments=arguments,
        )
        entry = entries.get(name)
        if entry is not None and entry.routes:
            routed = entry.route(tool_call_mode(entry.view, arguments))
            if routed is None or tool_route_unknown_arguments(
                entry.view, routed.capability, arguments
            ):
                entry = None
            else:
                targets[child.id] = replace(
                    child,
                    arguments=tool_route_arguments(
                        entry.view, routed.capability, arguments
                    ),
                )
                entry = routed
        facts = {
            "status": "admitted",
            "parent_call_id": parent.id,
            "sequence": sequence,
            "tool_name": name,
            "arguments": child.arguments,
            "capability_id": None if entry is None else entry.capability.id,
            "executor_id": None if entry is None else entry.executor_id,
            "contract_digest": (
                None if entry is None else entry.origin_revision_digest
            ),
            "sensitivity": sensitivity.value,
            "dispatched": False,
            "started_at": runtime._clock().isoformat(),
            "input_schema_digest": (
                None
                if entry is None
                else entry.input_schema_digest
                or "sha256:"
                + sha256(
                    canonical_json(entry.capability.input_schema).encode()
                ).hexdigest()
            ),
        }
        return child, entry, facts

    async def committed_capture(child: ToolCall) -> ArtifactRef | None:
        if runtime._artifacts is None:
            return None
        return next(
            (
                ref
                for ref in await runtime._artifacts.list_refs(
                    run_id=run.id, caller_principal_id=run.caller_principal_id
                )
                if ref.call_id == child.id
            ),
            None,
        )

    def capture_failure(
        child: ToolCall, original: ToolResultBlock, ref: ArtifactRef
    ) -> ToolResultBlock:
        reference = {
            key: value
            for key, value in artifact_ref_to_mapping(ref).items()
            if key != "provenance"
        }
        output = {
            "error": original.output.get(
                "error",
                {
                    "code": "analysis_capture_result_limited",
                    "message": "The export committed, but its metadata exceeded broker delivery bounds.",
                },
            ),
            "artifact": reference,
        }
        return replace(
            original,
            output=output,
            output_sha256=None,
            is_error=True,
            sensitivity=None,
            sensitivity_provenance={},
        )

    async def invoke(
        name: str,
        arguments: Mapping[str, object],
        sensitivity: ModelSensitivity,
        *,
        allocation=None,
        allowed: bool | None = None,
        recorded: bool = False,
        denial: str | None = None,
        result_bytes: int = 60_000,
        capture: bool = False,
    ) -> ProgrammaticReadResult:
        child, entry, facts = allocation or allocate(name, arguments, sensitivity)
        if allowed is None:
            allowed = session.evidence.count_tool_attempts(
                1, runtime._limits.max_tool_calls_per_run, programmatic=True
            )
        if not recorded:
            await session.writer.record_analysis(
                AnalysisEvidence(run.id, child.id, "child", facts)
            )
        captured_sensitivity: ModelSensitivity | None = None
        started = asyncio.get_running_loop().time()
        try:
            session.cancellation.raise_if_cancelled()
            if denial is not None:
                result = _error(
                    child,
                    denial,
                    "The broker request or its bounded delivery allowance is invalid; no reads were dispatched.",
                )
            elif not allowed or facts["sequence"] > child_limit:
                result = _error(
                    child,
                    "analysis_child_budget_exhausted",
                    "The shared run or cell child-call allowance is exhausted.",
                )
            elif entry is None or (
                not runtime._analysis_read_eligible(
                    entry.domain_owner_id, entry.capability
                )
                and not (
                    capture
                    and runtime._analysis_capture_eligible(
                        entry.domain_owner_id, entry.capability
                    )
                )
            ):
                result = _error(
                    child,
                    "analysis_child_not_admitted",
                    "The tool is outside this cell's frozen read surface.",
                )
            elif name not in await _domain_project(
                runtime._domains[entry.domain_owner_id], run, session, call=child
            ):
                result = _error(
                    child,
                    "analysis_child_revoked",
                    "The tool is no longer applicable.",
                )
            else:

                async def dispatched() -> None:
                    facts.update({"status": "running", "dispatched": True})
                    await session.writer.record_analysis(
                        AnalysisEvidence(run.id, child.id, "child", facts)
                    )

                result = await runtime._execute_one(
                    run,
                    targets.get(child.id, child),
                    entry,
                    sensitivity=sensitivity,
                    validated_arguments=None,
                    session=session,
                    dispatch_hook=dispatched,
                )
            if capture and (
                result.is_error
                or len(canonical_json(result.output).encode()) > result_bytes
            ):
                value = result.output.get("artifact")
                ref = (
                    artifact_ref_from_mapping(value)
                    if isinstance(value, Mapping)
                    else await committed_capture(child)
                )
                if ref is not None:
                    captured_sensitivity = max(
                        (ModelSensitivity(ref.sensitivity.value), sensitivity),
                        key=lambda value: value.routing_rank,
                    )
                    session.writer.retain_analysis_sensitivity(captured_sensitivity)
                    result = capture_failure(child, result, ref)
            if len(canonical_json(result.output).encode()) > result_bytes:
                original = result
                result = replace(
                    _error(
                        child,
                        "analysis_child_result_limited",
                        "The broker result exceeds its bounded delivery allowance; use a smaller read.",
                    ),
                    sensitivity=original.sensitivity,
                    sensitivity_provenance=original.sensitivity_provenance,
                )
            if result.sensitivity is not None:
                session.writer.retain_analysis_sensitivity(result.sensitivity)
            facts.update(
                {
                    "status": "failed" if result.is_error else "succeeded",
                    "result": result.output,
                    "result_digest": result.output_sha256
                    or "sha256:"
                    + sha256(canonical_json(result.output).encode()).hexdigest(),
                    "result_sensitivity": (
                        None
                        if (
                            classification := captured_sensitivity or result.sensitivity
                        )
                        is None
                        else classification.value
                    ),
                    "sensitivity_provenance": result.sensitivity_provenance,
                    "completed_at": runtime._clock().isoformat(),
                    "wall_seconds": asyncio.get_running_loop().time() - started,
                }
            )
            evidence = AnalysisEvidence(run.id, child.id, "child", facts)
            terminal_write = asyncio.create_task(
                session.writer.record_analysis(evidence)
            )
            cancelled = False
            while not terminal_write.done():
                try:
                    await asyncio.shield(terminal_write)
                except asyncio.CancelledError:
                    cancelled = True
            terminal_write.result()
            if cancelled:
                raise asyncio.CancelledError
            return ProgrammaticReadResult(
                {
                    "output": result.output,
                    "is_error": result.is_error,
                    "evidence_id": child.id,
                    "sensitivity": (
                        None
                        if (
                            classification := captured_sensitivity or result.sensitivity
                        )
                        is None
                        else classification.value
                    ),
                },
                evidence,
            )
        except BaseException as error:
            if isinstance(error, asyncio.CancelledError) and facts["status"] in {
                "succeeded",
                "failed",
            }:
                # State writes settle before reporting cancellation; preserve
                # the terminal observation instead of attempting an overwrite.
                raise

            async def settle_interruption() -> None:
                if capture and facts["dispatched"]:
                    ref = await committed_capture(child)
                    if ref is not None:
                        partial = capture_failure(
                            child,
                            _error(
                                child,
                                "analysis_capture_interrupted",
                                "The capture was interrupted after publication; the query was not replayed.",
                            ),
                            ref,
                        )
                        session.writer.retain_analysis_sensitivity(
                            max(
                                (
                                    ModelSensitivity(ref.sensitivity.value),
                                    sensitivity,
                                ),
                                key=lambda value: value.routing_rank,
                            )
                        )
                        facts.update(
                            result=partial.output,
                            result_digest="sha256:"
                            + sha256(
                                canonical_json(partial.output).encode()
                            ).hexdigest(),
                            result_sensitivity=(
                                None
                                if partial.sensitivity is None
                                else partial.sensitivity.value
                            ),
                            sensitivity_provenance=partial.sensitivity_provenance,
                        )
                await session.writer.record_analysis(
                    AnalysisEvidence(run.id, child.id, "child", facts)
                )

            facts.update(
                {
                    "status": "interrupted",
                    "wall_seconds": asyncio.get_running_loop().time() - started,
                }
            )
            task = asyncio.create_task(settle_interruption())
            await asyncio.shield(task)
            raise

    async def batch(
        calls: Sequence[object], sensitivity: ModelSensitivity, response_budget: int
    ) -> tuple[ProgrammaticReadResult, ...]:
        allowed = session.evidence.count_tool_attempts(
            len(calls), runtime._limits.max_tool_calls_per_run, programmatic=True
        )
        if not calls or len(calls) > 64:
            raise ValueError(
                "Analysis batches must contain a bounded nonempty call list"
            )
        malformed = any(
            not isinstance(item, Mapping)
            or set(item) != {"name", "arguments"}
            or not isinstance(item["name"], str)
            or not 0 < len(item["name"]) <= 256
            or not item["name"].strip()
            or not isinstance(item["arguments"], Mapping)
            for item in calls
        )
        normalized = tuple(
            (
                (
                    str(item["name"]),
                    (
                        cast(Mapping[str, object], item.get("arguments", {}))
                        if isinstance(item.get("arguments", {}), Mapping)
                        else {}
                    ),
                )
                if isinstance(item, Mapping)
                and isinstance(item.get("name"), str)
                and 0 < len(str(item["name"])) <= 256
                and str(item["name"]).strip()
                else ("<invalid-analysis-call>", {})
            )
            for item in calls
        )
        allocations = tuple(
            allocate(name, arguments, sensitivity) for name, arguments in normalized
        )
        allowed = allowed and sequence <= child_limit
        denial = "analysis_batch_invalid" if malformed else None
        if response_budget < len(calls) * 2_048:
            allowed = False
            denial = "analysis_batch_result_budget_exhausted"
        result_bytes = max(0, min(60_000, response_budget // len(calls) - 512))
        reserved = 0
        if allowed and denial is None:
            try:
                # Reserve every completion before creating an I/O task.
                for child, entry, facts in allocations:
                    await session.writer.record_analysis(
                        AnalysisEvidence(run.id, child.id, "child", facts)
                    )
                    reserved += 1
            except AnalysisTraceCapacityError:
                allowed = False
            except asyncio.CancelledError:

                async def settle_admissions():
                    for child, entry, facts in allocations:
                        facts.update(
                            status="interrupted",
                            interruption="batch_cancelled_before_dispatch",
                            wall_seconds=0.0,
                        )
                        await session.writer.record_analysis(
                            AnalysisEvidence(run.id, child.id, "child", facts)
                        )

                cleanup = asyncio.create_task(settle_admissions())
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError:
                        continue
                cleanup.result()
                raise
        if not allowed or denial is not None:
            return tuple(
                [
                    await invoke(
                        name,
                        arguments,
                        sensitivity,
                        allocation=allocation,
                        allowed=False,
                        recorded=index < reserved,
                        denial=denial,
                        result_bytes=result_bytes,
                    )
                    for index, ((name, arguments), allocation) in enumerate(
                        zip(normalized, allocations, strict=True)
                    )
                ]
            )
        read_gate = asyncio.Semaphore(runtime._limits.max_parallel_reads)
        source_gates: dict[str, asyncio.Semaphore] = {}

        async def read(name, arguments, allocation):
            key = _source_pressure_key(run, allocation[0])
            source_gate = source_gates.setdefault(
                key, asyncio.Semaphore(runtime._limits.max_parallel_reads_per_source)
            )
            async with read_gate:
                async with source_gate:
                    return await invoke(
                        name,
                        arguments,
                        sensitivity,
                        allocation=allocation,
                        allowed=True,
                        recorded=True,
                        result_bytes=result_bytes,
                    )

        tasks = tuple(
            asyncio.create_task(read(name, arguments, allocation))
            for (name, arguments), allocation in zip(
                normalized, allocations, strict=True
            )
        )
        analysis = cast("AnalysisCapabilityDomain", runtime._domains["analysis"])
        analysis.retain_child_tasks(run.id, tasks)
        try:
            await asyncio.wait(tasks)
        except BaseException:
            for task in tasks:
                if not task.done():
                    task.cancel("analysis_batch_interrupted")
            settled, pending = await _settle_cancelled_reads(
                tasks, timeout_seconds=runtime._side_effect_recovery_timeout_seconds
            )
            # Children cancelled while queued never reached invoke's handler.
            for task, (child, entry, facts) in zip(tasks, allocations, strict=True):
                if task in settled and facts["status"] == "admitted":
                    facts.update(
                        status="interrupted",
                        interruption="batch_cancelled_before_dispatch",
                        wall_seconds=0.0,
                    )
                    await session.writer.record_analysis(
                        AnalysisEvidence(run.id, child.id, "child", facts)
                    )
            for task in pending:
                task.add_done_callback(_consume_read_task)
            raise
        return tuple(task.result() for task in tasks)

    async def capture(
        name: str,
        arguments: Mapping[str, object],
        sensitivity: ModelSensitivity,
        denial: str | None,
    ) -> ProgrammaticReadResult:
        entry = entries.get(name)
        routed = (
            None
            if entry is None or not entry.routes
            else entry.route(tool_call_mode(entry.view, arguments))
        )
        # A capture installs one Arrow dataset: only an Arrow-producing
        # route of a capture-eligible capability qualifies.
        if (
            routed is None
            or entry is None
            or not runtime._analysis_capture_eligible(
                routed.domain_owner_id, routed.capability
            )
            or tool_route_arguments(entry.view, routed.capability, arguments).get(
                entry.view.mode_argument
            )
            != "arrow"
        ):
            denial = "analysis_capture_not_admitted"
        return await invoke(name, arguments, sensitivity, capture=True, denial=denial)

    return ProgrammaticReadBroker(invoke, batch, capture)
