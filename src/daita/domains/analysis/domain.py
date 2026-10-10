"""Own run-scoped interpreter state, current cell binding and cleanup."""

from __future__ import annotations

import asyncio
import os
import re
import secrets
import sys
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import cast

from ..._json import FrozenJsonObject, canonical_json
from ...adapters.analytical_workspace.native import NativePythonWorker, available
from ...adapters.analytical_workspace.outputs import (
    INPUT_MEDIA_TYPES,
    MEDIA_TYPES,
    VALIDATION_CODE,
    OutputCandidate,
    capture,
)
from ...adapters.analytical_workspace.runtime import runtime_status
from ...artifacts.models import ArtifactAuthorship, ArtifactDraft, ArtifactProvenance
from ...artifacts.store import AgentHomeArtifactStore
from ...capabilities import (
    AccessMode,
    ArtifactPolicy,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from ...capability_runtime import CapabilityFailure
from ...catalog.models import Sensitivity
from ...config import AnalysisLimits
from ...hosting.execution_governor import PermitLease, RunAdmissionCoordinator
from ...llm.models import ModelSensitivity, ToolCall
from ...loop.analysis import (
    AnalysisEvidence,
    AnalysisTraceCapacityError,
    ProgrammaticReadResult,
)
from ...loop.models import RunInput, RunOrigin
from ...loop.session import RunSession
from ...observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely
from ..data.export_capabilities import ArtifactCapabilityDomain

OWNER = "analysis"
CAPABILITY = "analysis.execute"


@dataclass(slots=True)
class _OwnedGeneration:
    worker: NativePythonWorker
    permit: PermitLease
    facts: dict[str, object]
    terminal: AnalysisEvidence | None = None
    persisted: bool = False
    admitted: bool = False


@dataclass(slots=True)
class _RunAnalysis:
    session: RunSession
    gate: asyncio.Lock = field(default_factory=asyncio.Lock)
    worker: NativePythonWorker | None = None
    generation: int = 0
    revision: int = 0
    records: list[dict[str, object]] = field(default_factory=list)
    parents: dict[str, Callable[..., Awaitable[ProgrammaticReadResult]]] = field(
        default_factory=dict
    )
    sensitivity: ModelSensitivity = ModelSensitivity.PUBLIC
    allocation: PermitLease | None = None
    candidates: dict[str, tuple[OutputCandidate, dict[str, object]]] = field(
        default_factory=dict
    )
    cells: list[dict[str, object]] = field(default_factory=list)
    children: list[dict[str, object]] = field(default_factory=list)
    parser_count: int = 0
    input_references: dict[str, Mapping[str, object]] = field(default_factory=dict)
    input_facts: list[dict[str, object]] = field(default_factory=list)
    admitted_artifacts: set[str] = field(default_factory=set)
    input_bytes: int = 0
    closed_cpu_seconds: float = 0.0
    charged_generations: set[str] = field(default_factory=set)
    owned: dict[str, _OwnedGeneration] = field(default_factory=dict)
    cleanup_blockers: set[str] = field(default_factory=set)
    authorities: dict[str, Callable[[ToolCall, ModelSensitivity], Awaitable[None]]] = (
        field(default_factory=dict)
    )
    admitted_calls: dict[str, ToolCall] = field(default_factory=dict)
    created_at: float = field(default_factory=lambda: asyncio.get_running_loop().time())
    artifact_bytes: int = 0
    worker_queue_wait_seconds: float = 0.0
    pending_cells: dict[str, dict[str, object]] = field(default_factory=dict)
    summary_admitted: bool = False
    max_run_calls: int = 64


class AnalysisCapabilityDomain:
    """One analysis capability, available only in local foreground sessions."""

    domain_owner_id = OWNER
    executor_id = "analysis.native_python"

    def __init__(
        self,
        coordinator: RunAdmissionCoordinator,
        evidence_owner: ArtifactCapabilityDomain | None = None,
        artifacts: AgentHomeArtifactStore | None = None,
        *,
        limits: AnalysisLimits = AnalysisLimits(),
        observer: AgentObserver | None = None,
    ) -> None:
        self._coordinator = coordinator
        self._evidence_owner = evidence_owner
        self._artifacts = artifacts
        self._limits = limits
        self._observer = observer
        description = (
            "Run normal Python for computation, simulation, text processing, structured-data validation, "
            "analysis and transformation of authenticated tool results in a private macOS sandbox. "
            "Variables persist only within this run. "
            "Use expected_state={generation:0,revision:0} initially; then use the returned state. "
            "tools.call(name, arguments) returns a dict: r['is_error'], r['output']['data'], r['evidence_id']. "
            "Register a file with outputs.add('name','file.png'); save_output selects one named output. "
            "Print bounded previews. NumPy, pandas, SciPy, PyArrow, Matplotlib, NetworkX and DuckDB are installed."
        )
        capability = Capability(
            id=CAPABILITY,
            description=description,
            executor_id=self.executor_id,
            input_schema={
                "type": "object",
                "properties": {
                    "code": {"type": "string", "maxLength": limits.code_bytes},
                    "expected_state": {
                        "type": "object",
                        "properties": {
                            "generation": {"type": "integer", "minimum": 0},
                            "revision": {"type": "integer", "minimum": 0},
                        },
                        "required": ["generation", "revision"],
                        "additionalProperties": False,
                    },
                    "timeout_seconds": {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "maximum": 30,
                    },
                },
                "required": ["code", "expected_state"],
                "additionalProperties": False,
            },
            output_kind="analysis.cell",
            output_schema={"type": "object"},
            access_mode=AccessMode.READ,
            artifact_policy=ArtifactPolicy(
                allowed_media_types=frozenset(MEDIA_TYPES.values()),
                allowed_authorships=frozenset(
                    {ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS}
                ),
                allowed_extensions=tuple(
                    (
                        media,
                        tuple(
                            extension
                            for extension, value in MEDIA_TYPES.items()
                            if value == media
                        ),
                    )
                    for media in sorted(set(MEDIA_TYPES.values()))
                ),
                artifact_required=False,
                max_artifact_count=1,
                max_bytes_per_artifact=limits.file_bytes,
                max_total_bytes_per_call=limits.file_bytes,
            ),
        )
        schema = dict(capability.input_schema)
        properties = dict(cast(Mapping[str, object], schema["properties"]))
        properties["save_output"] = {"type": "string", "minLength": 1, "maxLength": 64}
        properties["inputs"] = {
            "type": "object",
            "maxProperties": 16,
            "additionalProperties": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["tool_result", "artifact"]},
                    "call_id": {"type": "string", "minLength": 1, "maxLength": 256},
                    "artifact_id": {"type": "string", "minLength": 1, "maxLength": 256},
                },
                "required": ["kind"],
                "additionalProperties": False,
            },
        }
        schema["properties"] = properties
        output_properties = {
            "status": {
                "type": "string",
                "enum": ["success", "python_error", "state_lost"],
            },
            "kind": {"type": "string", "const": "cell"},
            "stdout": {"type": "string", "maxLength": 16384},
            "stderr": {"type": "string", "maxLength": 16384},
            "traceback": {"type": ["string", "null"], "maxLength": 8192},
            "expected_state": properties["expected_state"],
            "state_lost": {"type": "boolean"},
            "state_may_have_changed": {"type": "boolean"},
            "usage": {"type": "object"},
            "remaining_allowances": {"type": "object"},
            "cleanup": {"type": ["object", "null"]},
            "outputs": {"type": "array", "maxItems": 4, "items": {"type": "object"}},
            "child_calls": {
                "type": "array",
                "maxItems": 16,
                "items": {"type": "object"},
            },
            "failure": {"type": "string", "maxLength": 256},
            "save_status": {
                "type": "string",
                "enum": ["candidate_not_found", "requested", "failed"],
            },
            "save_error": {"type": "string", "maxLength": 256},
        }
        capability = replace(
            capability,
            input_schema=schema,
            output_schema={
                "type": "object",
                "properties": output_properties,
                "required": [
                    "status",
                    "stdout",
                    "stderr",
                    "traceback",
                    "expected_state",
                    "state_lost",
                    "state_may_have_changed",
                    "usage",
                    "outputs",
                ],
                "additionalProperties": False,
            },
        )
        self.declarations = CapabilityDeclarations(
            domain_owner_id=OWNER,
            capabilities=(capability,),
            executor_ids=(self.executor_id,),
            tool_views=(
                ToolView(
                    name="analysis_execute",
                    capability_id=CAPABILITY,
                    description=description,
                    presentation=ToolPresentation(
                        toolbox_id=ToolboxId.ANALYSIS,
                        load_mode=ToolLoadMode.ON_DEMAND,
                        text_trust=ToolTextTrust.CODE,
                        summary="Run Python locally for tabular and non-tabular computation.",
                        when_to_use=(
                            "Computation, simulation, text processing, structured-data validation, "
                            "transformation of authenticated tool results, cross-source analysis, "
                            "statistics, charts and datasets."
                        ),
                        keywords=(
                            "python",
                            "analysis",
                            "compute",
                            "duckdb",
                            "pandas",
                            "chart",
                            "computation",
                            "simulation",
                            "simulate",
                            "text processing",
                            "validation",
                            "transformation",
                            "transform",
                            "statistics",
                            "structured data validation",
                            "scipy",
                        ),
                    ),
                ),
            ),
        )
        self._runs: dict[str, _RunAnalysis] = {}

    def bind_parent(
        self,
        session: RunSession,
        run: RunInput,
        call: ToolCall,
        broker: Callable[..., Awaitable[ProgrammaticReadResult]],
        sensitivity: ModelSensitivity,
        authority: Callable[[ToolCall, ModelSensitivity], Awaitable[None]],
        max_run_calls: int,
    ) -> None:
        state = self._runs.get(run.id)
        if state is None:
            state = _RunAnalysis(session)
            self._runs[run.id] = state
            session.writer.bind_analysis_cleanup(lambda: self.close_run(run.id))
        if state.session.writer is not session.writer or call.id in state.parents:
            raise ValueError("Analysis parent identity was reused")
        state.parents[call.id] = broker
        state.max_run_calls = max_run_calls
        state.authorities[call.id] = authority
        state.sensitivity = max(
            state.sensitivity, sensitivity, key=lambda value: value.routing_rank
        )
        session.writer.retain_analysis_sensitivity(state.sensitivity)

    async def project(self, run: RunInput) -> tuple[str, ...]:
        return (
            ("analysis_execute",)
            if run.origin is RunOrigin.USER and available()
            else ()
        )

    def normalize_arguments(
        self, capability: Capability, arguments: Mapping[str, object]
    ) -> Mapping[str, object]:
        return arguments

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        raise CapabilityInputError(
            "analysis_session_required",
            "Analysis requires an admitted foreground run session.",
        )

    async def prepare_session_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
        session: RunSession,
    ) -> FrozenJsonObject:
        if run.origin is not RunOrigin.USER or session.run.id != run.id:
            raise CapabilityInputError(
                "analysis_session_required",
                "Analysis requires an admitted foreground run session.",
            )
        if len(str(arguments["code"]).encode()) > self._limits.code_bytes:
            raise CapabilityInputError(
                "analysis_code_limited",
                "Python source exceeds the configured cell byte allowance.",
            )
        if run.id not in self._runs:
            state = _RunAnalysis(session)
            self._runs[run.id] = state
            session.writer.bind_analysis_cleanup(lambda: self.close_run(run.id))
        state = self._runs[run.id]
        if not state.summary_admitted:
            await session.writer.record_analysis(
                AnalysisEvidence(
                    run.id,
                    "run-summary",
                    "run",
                    {
                        "status": "admitted",
                        "trace_bytes": self._limits.trace_bytes,
                        "trace_records": self._limits.trace_records,
                    },
                )
            )
            state.summary_admitted = True
        return arguments

    async def execute(self, request: ToolExecution) -> ToolOutput:
        state = self._runs[request.run_id]
        async with state.gate:
            cell_started = asyncio.get_running_loop().time()
            cell_deadline = min(
                state.session.absolute_deadline,
                cell_started
                + min(
                    self._limits.cell_seconds,
                    float(
                        cast(
                            float,
                            request.arguments.get(
                                "timeout_seconds", self._limits.cell_seconds
                            ),
                        )
                    ),
                ),
            )
            try:
                async with asyncio.timeout_at(cell_deadline):
                    return await self._execute_cell(
                        request, state, cell_deadline, cell_started
                    )
            except TimeoutError:
                await self._close_worker(state)
                facts = state.pending_cells.pop(request.call_id, None)
                if facts is not None:
                    await state.session.writer.record_analysis(
                        AnalysisEvidence(
                            request.run_id,
                            request.call_id,
                            "cell",
                            {
                                **facts,
                                "status": "interrupted",
                                "interruption": "cell_deadline",
                                "completion_observed": False,
                            },
                        )
                    )
                last = state.records[-1] if state.records else {}
                return ToolOutput(
                    kind="analysis.cell",
                    data={
                        "status": "state_lost",
                        "stdout": "",
                        "stderr": "",
                        "traceback": None,
                        "failure": "TimeoutError",
                        "state_lost": True,
                        "state_may_have_changed": True,
                        "expected_state": {
                            "generation": state.generation,
                            "revision": state.revision,
                        },
                        "usage": last.get("usage", {}),
                        "cleanup": last.get("cleanup"),
                        "outputs": [],
                    },
                )

    async def _execute_cell(
        self,
        request: ToolExecution,
        state: _RunAnalysis,
        cell_deadline: float,
        cell_started: float,
    ) -> ToolOutput:
        expected = dict(cast(Mapping[str, object], request.arguments["expected_state"]))
        if expected != {"generation": state.generation, "revision": state.revision}:
            raise CapabilityInputError(
                "analysis_stale_state",
                "Use the current host-issued generation and revision.",
            )
        state.session.cancellation.raise_if_cancelled()
        self._remaining_cpu(state)
        await state.session.writer.record_analysis(
            AnalysisEvidence(
                request.run_id,
                request.call_id,
                "cell",
                {
                    "status": "admitted",
                    "code": request.arguments["code"],
                    "generation": state.generation,
                    "revision": state.revision,
                    "sensitivity": state.sensitivity.value,
                },
            )
        )
        state.pending_cells[request.call_id] = {
            "code": request.arguments["code"],
            "generation": state.generation,
            "sensitivity": state.sensitivity.value,
        }
        if state.worker is None:
            queued_at = asyncio.get_running_loop().time()
            try:
                state.allocation = await self._coordinator.analysis_worker_permit(
                    deadline=cell_deadline,
                    cancellation=state.session.cancellation,
                )
            finally:
                state.worker_queue_wait_seconds += (
                    asyncio.get_running_loop().time() - queued_at
                )
            if (
                state.closed_cpu_seconds >= self._limits.cpu_seconds
                or state.generation >= self._limits.generations
            ):
                state.session.evidence.analysis_budget_exhausted = True
                raise RuntimeError("The cumulative analytical allowance is exhausted")
            state.worker = NativePythonWorker(
                durable_closure=True,
                limits=replace(
                    self._limits,
                    cpu_seconds=self._limits.cpu_seconds - state.closed_cpu_seconds,
                ),
            )
            state.generation += 1
            assert state.allocation is not None
            state.owned[f"generation-{state.generation}"] = _OwnedGeneration(
                state.worker, state.allocation, {"generation": state.generation}
            )
            await state.session.writer.record_analysis(
                AnalysisEvidence(
                    request.run_id,
                    f"generation-{state.generation}",
                    "generation",
                    {
                        "status": "admitted",
                        "generation": state.generation,
                        **state.worker.identity,
                        "memory_reservation_bytes": state.worker.limits.memory_bytes,
                        "scratch_reservation_bytes": state.worker.limits.scratch_bytes,
                    },
                )
            )
            state.owned[f"generation-{state.generation}"].admitted = True
            try:
                await state.worker.start()
            except BaseException:
                await self._close_worker(state)
                await state.session.writer.record_analysis(
                    AnalysisEvidence(
                        request.run_id,
                        request.call_id,
                        "cell",
                        {
                            "status": "interrupted",
                            "code": request.arguments["code"],
                            "generation": state.generation,
                            "sensitivity": state.sensitivity.value,
                            "interruption": "worker_start_failed",
                        },
                    )
                )
                state.pending_cells.pop(request.call_id, None)
                raise
            await state.session.writer.record_analysis(
                AnalysisEvidence(
                    request.run_id,
                    f"generation-{state.generation}",
                    "generation",
                    {
                        "status": "running",
                        "generation": state.generation,
                        "pid": state.worker.worker_pid,
                        **state.worker.identity,
                        "usage": dict(state.worker.usage),
                    },
                )
            )
        worker = state.worker

        async def revalidate() -> None:
            state.session.cancellation.raise_if_cancelled()
            if (
                runtime_status()["runtime_identity"]
                != worker.runtime["runtime_identity"]
            ):
                raise RuntimeError("The installed analytical runtime changed")
            for admitted in state.admitted_calls.values():
                await state.authorities[request.call_id](admitted, state.sensitivity)
            if self._artifacts is not None:
                for artifact_id in state.admitted_artifacts:
                    if await self._artifacts.find_ref(artifact_id) is None:
                        raise RuntimeError(
                            "An admitted analytical artifact was deleted"
                        )

        worker.revalidate = revalidate

        def observe(usage: Mapping[str, object]) -> None:
            _emit_safely(
                self._observer,
                AgentEvent(
                    kind=AgentEventKind.ANALYSIS_UPDATED,
                    occurred_at=datetime.now(timezone.utc),
                    run_id=request.run_id,
                    conversation_id=request.conversation_id or request.run_id,
                    data=FrozenJsonObject.from_mapping(
                        {"generation": state.generation, **dict(usage)}
                    ),
                ),
            )

        worker.observe = observe
        worker.limits = replace(
            worker.limits,
            cpu_seconds=max(0.001, self._limits.cpu_seconds - state.closed_cpu_seconds),
        )
        child_start = len(state.children)

        async def broker(
            name: str, arguments: Mapping[str, object]
        ) -> dict[str, object]:
            await revalidate()
            read = await state.parents[request.call_id](
                name, arguments, state.sensitivity
            )
            result = read.response
            facts = read.evidence.facts
            envelope = cast(Mapping[str, object], result["output"])
            data = envelope.get("data", {})
            data = data if isinstance(data, Mapping) else {}
            classification = result.get("sensitivity")
            if classification is not None:
                state.sensitivity = max(
                    state.sensitivity,
                    ModelSensitivity(str(classification)),
                    key=lambda value: value.routing_rank,
                )
                state.session.writer.retain_analysis_sensitivity(state.sensitivity)
            state.children.append(
                {
                    "evidence_id": result["evidence_id"],
                    "tool_name": name,
                    "is_error": result["is_error"],
                    "sensitivity": classification,
                    "capability_id": facts["capability_id"],
                    "executor_id": facts["executor_id"],
                    "contract_digest": facts["contract_digest"],
                    "input_schema_digest": facts.get("input_schema_digest"),
                    "arguments_sha256": "sha256:"
                    + sha256(canonical_json(arguments).encode()).hexdigest(),
                    "coverage": {
                        key: data.get(key)
                        for key in (
                            "complete",
                            "truncated",
                            "row_count",
                            "returned_count",
                        )
                    },
                    "source_metadata": {
                        key: data[key]
                        for key in (
                            "source_id",
                            "source_revision",
                            "resource_id",
                            "resource_revision",
                            "resource_revisions",
                            "revision",
                        )
                        if key in data
                    },
                    "result_sha256": "sha256:"
                    + sha256(canonical_json(result["output"]).encode()).hexdigest(),
                }
            )
            if result["is_error"] is False:
                state.admitted_calls[str(result["evidence_id"])] = ToolCall(
                    id=str(result["evidence_id"]), name=name, arguments=arguments
                )
            return result

        async def compute() -> PermitLease:
            return await self._compute_permit(state, worker, cell_deadline)

        try:
            current_run = replace(
                state.session.run, resolved_source_scope=request.source_scope
            )
            current_call = ToolCall(
                id=request.call_id,
                name="analysis_execute",
                arguments=request.arguments,
            )
            requested_inputs = request.arguments.get("inputs", {})
            if not isinstance(requested_inputs, Mapping):
                raise ValueError("Input references must be an object")
            input_manifest = await self._bind_inputs(
                state,
                current_run,
                current_call,
                requested_inputs,
                compute,
                cell_deadline,
            )
            await revalidate()
            result = await worker.execute(
                str(request.arguments["code"]),
                deadline=cell_deadline,
                broker=broker,
                compute=compute,
                inputs=input_manifest,
            )
            state.revision += 1
            state.cells.append(
                {
                    "call_id": request.call_id,
                    "code": request.arguments["code"],
                    "code_sha256": "sha256:"
                    + sha256(str(request.arguments["code"]).encode()).hexdigest(),
                }
            )
            for key in ("stdout", "stderr", "traceback"):
                value = result.get(key)
                if isinstance(value, str):
                    for path in (
                        str(worker.scratch),
                        str(
                            Path(__file__).resolve().parents[2]
                            / "adapters"
                            / "analytical_workspace"
                            / "worker.py"
                        ),
                        sys.prefix,
                        sys.base_prefix,
                    ):
                        value = value.replace(path, "<runtime>")
                    result[key] = value
            captured = (
                capture(
                    worker.scratch,
                    result.pop("outputs", []),
                    max_bytes=self._limits.file_bytes,
                    max_count=self._limits.outputs_per_cell,
                )
                if result["status"] == "success"
                else ()
            )
            await revalidate()
            new_candidates = [
                candidate
                for candidate in captured
                if candidate.name not in state.candidates
            ]
            if (
                len(state.candidates) + len(new_candidates)
                > self._limits.outputs_per_run
                or sum(
                    len(candidate.content) for candidate, _ in state.candidates.values()
                )
                + sum(len(candidate.content) for candidate in new_candidates)
                > self._limits.output_bytes_per_run
            ):
                raise ValueError("Run output snapshot allowance exhausted")
            for candidate in captured:
                if candidate.name in state.candidates:
                    existing, _ = state.candidates[candidate.name]
                    if (
                        candidate.sha256 != existing.sha256
                        or candidate.filename != existing.filename
                    ):
                        raise ValueError(
                            "Output candidate names are immutable within a run"
                        )
                    continue
                await self._validate_candidate(state, candidate, compute, cell_deadline)
                await revalidate()
                code_bytes = sum(
                    len(str(cell["code"]).encode()) for cell in state.cells
                )
                state.candidates[candidate.name] = (
                    candidate,
                    {
                        "authority": "host_native_computation",
                        "generation": state.generation,
                        "runtime_identity": worker.runtime["runtime_identity"],
                        "run_id": request.run_id,
                        "producing_call_id": request.call_id,
                        "output_sha256": candidate.sha256,
                        "children": list(state.children),
                        "inputs": list(state.input_facts),
                        "cells": (
                            list(state.cells)
                            if code_bytes <= 128 * 1024
                            else [
                                {
                                    "call_id": cell["call_id"],
                                    "code_sha256": cell["code_sha256"],
                                }
                                for cell in state.cells
                            ]
                        ),
                        "source_available": code_bytes <= 128 * 1024,
                        "inputs_available": False,
                        "reproduction": "Raw inputs are not retained; source may be bounded.",
                    },
                )
            result["outputs"] = [
                {
                    "name": candidate.name,
                    "filename": candidate.filename,
                    "sha256": candidate.sha256,
                    "byte_size": len(candidate.content),
                }
                for candidate in captured
            ]
            result.update(
                {
                    "expected_state": {
                        "generation": state.generation,
                        "revision": state.revision,
                    },
                    "state_may_have_changed": result["status"] == "python_error",
                    "state_lost": False,
                    "usage": {
                        **worker.usage,
                        "cell_wall_seconds": asyncio.get_running_loop().time()
                        - cell_started,
                    },
                }
            )
            result["child_calls"] = [
                {
                    "evidence_id": child["evidence_id"],
                    "tool_name": child["tool_name"],
                    "is_error": child["is_error"],
                }
                for child in state.children[child_start:]
            ]
            cpu = (
                None
                if worker.usage["user_cpu_seconds"] is None
                or worker.usage["system_cpu_seconds"] is None
                else worker.usage["user_cpu_seconds"]
                + worker.usage["system_cpu_seconds"]
            )
            result["remaining_allowances"] = {
                "run_cpu_seconds": (
                    None
                    if cpu is None
                    else max(
                        0.0, self._limits.cpu_seconds - state.closed_cpu_seconds - cpu
                    )
                ),
                "cell_wall_seconds": max(
                    0.0, cell_deadline - asyncio.get_running_loop().time()
                ),
                "run_wall_seconds": max(
                    0.0,
                    state.session.absolute_deadline - asyncio.get_running_loop().time(),
                ),
                "run_tool_calls": max(
                    0, state.max_run_calls - state.session.evidence.tool_calls_attempted
                ),
                "cell_child_calls": max(
                    0, self._limits.child_calls - len(state.children[child_start:])
                ),
                "input_bytes": self._limits.input_bytes - state.input_bytes,
                "output_bytes": self._limits.output_bytes_per_run
                - sum(
                    len(candidate.content) for candidate, _ in state.candidates.values()
                ),
            }
            await state.session.writer.record_analysis(
                AnalysisEvidence(
                    request.run_id,
                    request.call_id,
                    "cell",
                    {
                        "status": "completed",
                        "code": request.arguments["code"],
                        "generation": state.generation,
                        "revision": state.revision,
                        "sensitivity": state.sensitivity.value,
                        "result": result,
                        "inputs": list(state.input_facts),
                    },
                )
            )
            state.pending_cells.pop(request.call_id, None)
            draft = None
            save = request.arguments.get("save_output")
            if save is not None:
                selected = state.candidates.get(str(save))
                if selected is None:
                    result["save_status"] = "candidate_not_found"
                else:
                    candidate, provenance = selected
                    draft = ArtifactDraft(
                        content=candidate.content,
                        suggested_filename=candidate.filename,
                        media_type=candidate.media_type,
                        sensitivity=Sensitivity(state.sensitivity.value),
                        provenance=ArtifactProvenance(
                            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
                        ),
                        computation_evidence=provenance,
                    )
                    result["save_status"] = "requested"
            return ToolOutput(kind="analysis.cell", data=result, artifact=draft)
        except BaseException as error:

            async def settle_failure() -> None:
                try:
                    await self._close_worker(state)
                finally:
                    await state.session.writer.record_analysis(
                        AnalysisEvidence(
                            request.run_id,
                            request.call_id,
                            "cell",
                            {
                                "status": "interrupted",
                                "code": request.arguments["code"],
                                "generation": state.generation,
                                "sensitivity": state.sensitivity.value,
                                "usage": dict(worker.usage),
                                "cleanup": worker.cleanup,
                                "inputs": list(state.input_facts),
                            },
                        )
                    )
                    state.pending_cells.pop(request.call_id, None)

            task = asyncio.create_task(settle_failure())
            await asyncio.shield(task)
            if not isinstance(error, Exception):
                raise
            return ToolOutput(
                kind="analysis.cell",
                data={
                    "status": "state_lost",
                    "stdout": "",
                    "stderr": "",
                    "traceback": None,
                    "failure": type(error).__name__,
                    "state_lost": True,
                    "state_may_have_changed": True,
                    "expected_state": {
                        "generation": state.generation,
                        "revision": state.revision,
                    },
                    "usage": dict(worker.usage),
                    "cleanup": worker.cleanup,
                    "outputs": [],
                },
            )
        finally:
            state.parents.pop(request.call_id, None)

    def _remaining_cpu(self, state: _RunAnalysis) -> float:
        if any(run.cleanup_blockers for run in self._runs.values()):
            raise RuntimeError("Unresolved analytical closure blocks admission")
        used = state.closed_cpu_seconds
        for key, owner in state.owned.items():
            if key in state.charged_generations:
                continue
            worker = owner.worker
            if worker.process is None:
                continue  # Still host-owned, before any native launch.
            worker.sample()
            user, system = (
                worker.usage["user_cpu_seconds"],
                worker.usage["system_cpu_seconds"],
            )
            if user is None or system is None:
                raise RuntimeError(
                    "Cumulative analytical CPU measurement is unavailable"
                )
            used += float(user) + float(system)
        remaining = self._limits.cpu_seconds - used
        if remaining <= 0:
            state.session.evidence.analysis_budget_exhausted = True
            state.session.writer.analysis_budget_exhausted = True
            raise RuntimeError("Cumulative analytical CPU allowance exhausted")
        return remaining

    async def _compute_permit(
        self, state: _RunAnalysis, worker: NativePythonWorker, deadline: float
    ) -> PermitLease:
        permit = await self._coordinator.analysis_compute_permit(
            deadline=deadline,
            cancellation=state.session.cancellation,
        )
        try:
            state.session.cancellation.raise_if_cancelled()
            if asyncio.get_running_loop().time() >= deadline:
                raise TimeoutError("Analytical compute deadline exhausted")
            remaining = self._remaining_cpu(state)
            own_cpu = float(worker.usage["user_cpu_seconds"]) + float(
                worker.usage["system_cpu_seconds"]
            )
            # Native enforcement is lifetime cumulative. Own CPU is counted once
            # in the run total, and added back only to express that lifetime cap.
            worker.limits = replace(worker.limits, cpu_seconds=own_cpu + remaining)
            return permit
        except BaseException:
            await permit.release()
            raise

    async def _settle_generation(self, state: _RunAnalysis, key: str) -> None:
        owner = state.owned[key]
        worker = owner.worker
        cleanup = await worker.close()
        measured = worker.usage.get("cpu_complete") is True
        if key not in state.charged_generations and measured:
            state.closed_cpu_seconds += float(worker.usage["user_cpu_seconds"]) + float(
                worker.usage["system_cpu_seconds"]
            )
            state.charged_generations.add(key)
        if state.closed_cpu_seconds >= self._limits.cpu_seconds:
            state.session.evidence.analysis_budget_exhausted = True
            state.session.writer.analysis_budget_exhausted = True
        complete = measured and all(
            cleanup.get(name) is True
            for name in (
                "process_reaped",
                "scratch_deleted",
                "descriptors_closed",
                "child_io_settled",
            )
        )
        if owner.terminal is None:
            owner.terminal = AnalysisEvidence(
                state.session.run.id,
                key,
                "generation",
                {
                    "status": "closed" if complete else "cleanup_failed",
                    **owner.facts,
                    "pid": worker.worker_pid,
                    **worker.identity,
                    "usage": dict(worker.usage),
                    "cleanup": cleanup,
                },
            )
            state.records.append(dict(owner.terminal.facts))
        if not complete:
            state.cleanup_blockers.add(key)
        try:
            if not owner.persisted:
                try:
                    await state.session.writer.record_analysis(owner.terminal)
                    owner.persisted = True
                except AnalysisTraceCapacityError:
                    if not owner.admitted and worker.process is None and complete:
                        # The short admission transaction positively rejected
                        # this allocation before dispatch or durable ownership.
                        worker.dispose_closure_evidence()
                        await owner.permit.release()
                        del state.owned[key]
                    raise
            if not complete:
                raise RuntimeError(
                    "Native closure is incomplete; replacement and successful acceptance are blocked"
                )
            worker.dispose_closure_evidence()
            disposed = replace(
                owner.terminal, facts={**owner.terminal.facts, "proof_disposed": True}
            )
            await state.session.writer.record_analysis(disposed)
            owner.terminal = disposed
            await owner.permit.release()
            del state.owned[key]
        except BaseException:
            state.cleanup_blockers.add(key)
            raise

    async def _close_worker(self, state: _RunAnalysis) -> None:
        if state.worker is not None:
            await self._settle_generation(state, f"generation-{state.generation}")
            state.worker = None
            state.allocation = None
            state.revision = 0
            state.input_references.clear()
        elif state.allocation is not None:
            await state.allocation.release()
            state.allocation = None

    async def _validate_candidate(
        self,
        state: _RunAnalysis,
        candidate: OutputCandidate,
        compute: Callable[[], Awaitable[PermitLease]],
        cell_deadline: float,
    ) -> None:
        remaining = self._remaining_cpu(state)
        permit = await self._coordinator.analysis_parser_permit(
            deadline=cell_deadline,
            cancellation=state.session.cancellation,
        )
        try:
            # Waiting for a parser lane may have consumed the deadline or changed
            # admission. Recheck before allocation and before every resume.
            state.session.cancellation.raise_if_cancelled()
            remaining = self._remaining_cpu(state)
            parser = NativePythonWorker(
                durable_closure=True,
                limits=replace(self._limits, cpu_seconds=remaining),
            )
        except BaseException:
            await permit.release()
            raise
        state.parser_count += 1
        key = f"parser-{state.parser_count}"
        state.owned[key] = _OwnedGeneration(parser, permit, {"role": "output_parser"})
        try:
            await state.session.writer.record_analysis(
                AnalysisEvidence(
                    state.session.run.id,
                    key,
                    "generation",
                    {
                        "status": "admitted",
                        "role": "output_parser",
                        **parser.identity,
                    },
                )
            )
            state.owned[key].admitted = True
            written = (parser.scratch / "candidate.bin").write_bytes(candidate.content)
            parser.usage["input_bytes"] = written
            parser.usage["validation_input_bytes"] = written
            await parser.start()
            parser_deadline = min(
                cell_deadline,
                asyncio.get_running_loop().time() + self._limits.parser_seconds,
            )
            result = await parser.execute(
                VALIDATION_CODE[Path(candidate.filename).suffix]
                .replace("134217728", str(self._limits.decoded_bytes))
                .replace("1000000", str(self._limits.table_rows))
                .replace("256", str(self._limits.table_columns)),
                deadline=parser_deadline,
                compute=lambda: self._compute_permit(state, parser, parser_deadline),
            )
            if result["status"] != "success":
                raise ValueError("Output candidate format validation failed")
        finally:
            task = asyncio.create_task(self._settle_generation(state, key))
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise

    async def _bind_inputs(
        self,
        state: _RunAnalysis,
        run: RunInput,
        call: ToolCall,
        requested: Mapping[str, object],
        compute: Callable[[], Awaitable[PermitLease]],
        cell_deadline: float,
    ) -> dict[str, object]:
        references = dict(state.input_references)
        for name, value in requested.items():
            if re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_-]{0,63}", name
            ) is None or not isinstance(value, Mapping):
                raise ValueError(
                    "Inputs require bounded logical names and exact references"
                )
            kind = value.get("kind")
            expected = (
                {"kind", "call_id"}
                if kind == "tool_result"
                else {"kind", "artifact_id"}
            )
            if kind not in {"tool_result", "artifact"} or set(value) != expected:
                raise ValueError("Unknown input reference contract")
            references[name] = value
        if len(references) > 16:
            raise ValueError("Run input reference allowance exhausted")
        if not references:
            return {}
        manifest: dict[str, object] = {}
        assert state.worker is not None
        for name, reference in references.items():
            kind = reference["kind"]
            if kind == "tool_result":
                if self._evidence_owner is None:
                    raise ValueError("Authenticated input evidence is unavailable")
                evidence = (
                    await self._evidence_owner.authenticate_analysis_inputs(
                        run, call, (str(reference["call_id"]),)
                    )
                )[0]
                if evidence.capability.access_mode is not AccessMode.READ and not (
                    evidence.capability.access_mode is AccessMode.NONE
                    and evidence.capability.id
                    in self._evidence_owner.programmatic_read_capability_ids
                ):
                    raise ValueError("Analysis inputs require read result evidence")
                assert evidence.block.sensitivity is not None
                state.admitted_calls[evidence.call.id] = evidence.call
                await state.authorities[call.id](evidence.call, state.sensitivity)
                state.sensitivity = max(
                    state.sensitivity,
                    evidence.block.sensitivity,
                    key=lambda value: value.routing_rank,
                )
                content = canonical_json(evidence.block.output).encode()
                media_type, extension = "application/json", ".json"
                facts = {
                    "call_id": evidence.call.id,
                    "result_sha256": evidence.block.output_sha256,
                    "coverage": {
                        "truncated": evidence.data.get("truncated"),
                        "complete": evidence.data.get("complete"),
                    },
                }
            else:
                if self._artifacts is None:
                    raise ValueError("Artifact input storage is unavailable")
                ref = await self._artifacts.find_ref(str(reference["artifact_id"]))
                if (
                    ref is None
                    or ref.byte_size > self._limits.file_bytes
                    or Path(ref.filename).suffix not in INPUT_MEDIA_TYPES
                ):
                    raise ValueError(
                        "Artifact input is missing or outside the admitted formats and bounds"
                    )
                payload = await self._artifacts.read_ref(ref)
                content, media_type, extension = (
                    payload.content,
                    ref.media_type,
                    Path(ref.filename).suffix,
                )
                state.sensitivity = max(
                    state.sensitivity,
                    ModelSensitivity(ref.sensitivity.value),
                    key=lambda value: value.routing_rank,
                )
                facts = {
                    "artifact_id": ref.artifact_id,
                    "result_sha256": ref.sha256,
                    "coverage": "retained_artifact",
                }
                state.admitted_artifacts.add(ref.artifact_id)
                if name in requested:
                    await self._validate_candidate(
                        state,
                        OutputCandidate(
                            name, ref.filename, media_type, content, ref.sha256
                        ),
                        compute,
                        cell_deadline,
                    )
            state.session.writer.retain_analysis_sensitivity(state.sensitivity)
            if name not in requested:
                continue
            if state.input_bytes + len(content) > self._limits.input_bytes:
                raise ValueError("Run captured-input byte allowance exhausted")
            filename = f"input-{secrets.token_hex(16)}{extension}"
            descriptor = os.open(
                state.worker.scratch / filename,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                0o600,
            )
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
            state.input_bytes += len(content)
            state.worker.usage["input_bytes"] += len(content)
            manifest[name] = {"path": filename, "kind": kind, "media_type": media_type}
            input_fact = {
                "name": name,
                **facts,
                "byte_size": len(content),
                "conversion": "none",
            }
            if input_fact not in state.input_facts:
                state.input_facts.append(input_fact)
        state.input_references = references
        return manifest

    async def close_run(self, run_id: str) -> None:
        state = self._runs.get(run_id)
        if state is None:
            return
        async with state.gate:
            failures = []
            for key in tuple(state.owned):
                try:
                    await self._settle_generation(state, key)
                    if key.startswith("generation-"):
                        state.worker = None
                        state.allocation = None
                except Exception as error:
                    failures.append(error)
            if failures or state.cleanup_blockers:
                raise RuntimeError(
                    "Unresolved analytical closure prevents terminal success"
                ) from (failures[0] if failures else None)
            await self._close_worker(state)
            for call_id, facts in state.pending_cells.items():
                await state.session.writer.record_analysis(
                    AnalysisEvidence(
                        run_id,
                        call_id,
                        "cell",
                        {
                            **facts,
                            "status": "interrupted",
                            "interruption": "run_closed_before_dispatch",
                            "completion_observed": False,
                        },
                    )
                )
            state.pending_cells.clear()
            await state.session.writer.record_analysis(
                AnalysisEvidence(
                    run_id,
                    "run-summary",
                    "run",
                    {
                        "status": "closed",
                        "trace_bytes": self._limits.trace_bytes,
                        "trace_records": self._limits.trace_records,
                        "wall_seconds": (
                            None
                            if state.session.evidence.run_started_monotonic is None
                            else asyncio.get_running_loop().time()
                            - state.session.evidence.run_started_monotonic
                        ),
                        "analysis_wall_seconds": asyncio.get_running_loop().time()
                        - state.created_at,
                        "worker_queue_wait_seconds": state.worker_queue_wait_seconds,
                        "tool_calls_attempted": state.session.evidence.tool_calls_attempted,
                        "outer_tool_calls": state.session.evidence.outer_tool_counts(),
                        "child_attempted": state.session.evidence.programmatic_tool_calls_attempted,
                        "input_bytes": state.input_bytes,
                        "candidate_bytes": sum(
                            len(value[0].content) for value in state.candidates.values()
                        ),
                        "artifact_bytes_committed": state.artifact_bytes,
                        "reservations_released": state.allocation is None,
                    },
                )
            )
        del self._runs[run_id]

    async def handoff_recovery(self) -> None:
        """Transfer terminal, physically stopped owners to durable startup recovery.

        Composition calls this only after closing host admission. No lane can
        become reusable in that coordinator. Live/native-uncertain owners must
        still pass its drain barrier, and keep the home writer lease otherwise.
        """
        for state in self._runs.values():
            if state.session.writer.state.value != "terminal":
                continue
            for owner in state.owned.values():
                cleanup = owner.worker.cleanup or {}
                if owner.persisted and all(
                    cleanup.get(name) is True
                    for name in (
                        "process_reaped",
                        "descriptors_closed",
                        "child_io_settled",
                    )
                ):
                    await owner.permit.release()

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> ToolOutput:
        state = self._runs[run.id]
        if state.worker is not None and state.worker.revalidate is not None:
            try:
                await state.worker.revalidate()
            except BaseException:
                await self._close_worker(state)
                raise
        return replace(
            output,
            sensitivity=max(
                request_sensitivity,
                self._runs[run.id].sensitivity,
                key=lambda value: value.routing_rank,
            ),
            sensitivity_provenance={
                "authority": "host_analysis_inputs",
                "run_id": run.id,
                "call_id": call.id,
            },
        )

    def artifact_committed(self, run_id: str, byte_size: int) -> None:
        self._runs[run_id].artifact_bytes += byte_size

    @property
    def child_call_limit(self) -> int:
        return self._limits.child_calls

    def artifact_save_failed(self, output: ToolOutput, code: str) -> ToolOutput:
        return replace(
            output,
            artifact=None,
            data={**dict(output.data), "save_status": "failed", "save_error": code},
        )

    def normalize_error(
        self, call: ToolCall, error: BaseException
    ) -> CapabilityFailure | None:
        if isinstance(error, (RuntimeError, TimeoutError)):
            return CapabilityFailure(
                "analysis_worker_failed",
                "The native worker was closed; its interpreter state is lost.",
            )
        return None

    async def prepare_automation_grant(self, *args, **kwargs):
        raise ValueError("Analysis is foreground-only")

    async def side_effect_plan(self, *args, **kwargs):
        raise ValueError("Analysis has no external effects")
