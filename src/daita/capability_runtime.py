"""Validate, govern, execute, and normalize capabilities across static domains."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import Enum
from hashlib import sha256
from typing import Protocol, cast

from ._json import FrozenJsonObject, canonical_json
from .artifacts.models import (
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
    OperationalEffect,
    SideEffectExecutor,
    ToolExecution,
    ToolExposureClass,
    ToolOutput,
    ToolOutputValidationError,
    ToolView,
    validate_tool_schema_value,
)
from .errors import DaitaError
from .llm.errors import (
    ToolCatalogLimitExceeded,
    ToolManifestLimitExceeded,
    ToolSurfaceLimitExceeded,
)
from .llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelSensitivity,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
)
from .loop.models import (
    LoopLimits,
    RunInput,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolBatchOutcome,
    ToolProjectionMode,
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


@dataclass(frozen=True, slots=True)
class InternalCapabilityRequest:
    """Trusted code-owned execution request for one internal-only capability."""

    run: RunInput
    call_id: str
    capability_id: str
    contract_digest: str
    arguments: Mapping[str, object]
    sensitivity: ModelSensitivity
    reserved_artifact_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.run, RunInput):
            raise TypeError("internal request run must be RunInput")
        for value, name in (
            (self.call_id, "internal call_id"),
            (self.capability_id, "internal capability_id"),
            (self.contract_digest, "internal contract_digest"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be non-empty text")
        if re.fullmatch(r"sha256:[0-9a-f]{64}", self.contract_digest) is None:
            raise ValueError("internal contract_digest must use sha256")
        if not isinstance(self.sensitivity, ModelSensitivity):
            raise TypeError("internal request sensitivity must be ModelSensitivity")
        if self.reserved_artifact_id is not None and (
            not isinstance(self.reserved_artifact_id, str)
            or not self.reserved_artifact_id.strip()
        ):
            raise ValueError("reserved_artifact_id must be non-empty text or None")
        object.__setattr__(
            self,
            "arguments",
            FrozenJsonObject.from_mapping(self.arguments),
        )


@dataclass(frozen=True, slots=True)
class InternalCapabilityOutcome:
    output: ToolOutput
    artifact_ref: ArtifactRef | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.output, ToolOutput):
            raise TypeError("internal outcome output must be ToolOutput")
        if self.artifact_ref is not None and not isinstance(
            self.artifact_ref, ArtifactRef
        ):
            raise TypeError("internal outcome artifact_ref must be ArtifactRef or None")


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
        *,
        request_sensitivity: ModelSensitivity,
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
        *,
        request_sensitivity: ModelSensitivity,
    ) -> ToolOutput: ...

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None: ...


class ToolInvocationMode(str, Enum):
    DIRECT = "direct"
    DEFERRED = "deferred"


@dataclass(frozen=True, slots=True)
class DomainToolManifestEntry:
    domain_owner_id: str
    summary: str
    direct_count: int
    deferred_count: int


@dataclass(frozen=True, slots=True)
class RunToolCatalogEntry:
    view: ToolView
    capability: Capability
    domain_owner_id: str
    executor_id: str
    input_schema_digest: str
    origin_revision_digest: str
    invocation_mode: ToolInvocationMode

    @property
    def parameter_names(self) -> tuple[str, ...]:
        properties = self.capability.input_schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return ()
        return tuple(sorted(name for name in properties if isinstance(name, str)))


@dataclass(frozen=True, slots=True)
class RunToolCatalog:
    run_id: str
    agent_id: str
    execution_scope_digest: str
    registry_digest: str
    catalog_digest: str
    entries: tuple[RunToolCatalogEntry, ...]
    domain_manifest: tuple[DomainToolManifestEntry, ...]
    provider_definitions: tuple[ToolDefinition, ...]
    aggregate_bytes: int
    manifest_bytes: int
    manifest_token_limit: int

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(entry.capability.id for entry in self.entries)

    @property
    def manifest_payload(self) -> tuple[FrozenJsonObject, ...]:
        return tuple(
            FrozenJsonObject.from_mapping(item)
            for item in _manifest_material(self.domain_manifest)
        )


@dataclass(frozen=True, slots=True)
class DeferredToolReference:
    tool_ref: str
    tool_name: str
    capability_id: str
    domain_owner_id: str
    executor_id: str
    input_schema_digest: str
    origin_revision_digest: str
    catalog_digest: str
    description_bytes: int


@dataclass(frozen=True, slots=True)
class StepToolProjection:
    run_id: str
    catalog_digest: str
    projection_digest: str
    provider_definitions: tuple[ToolDefinition, ...]
    catalog_entries: tuple[RunToolCatalogEntry, ...]
    direct_resolution_entries: tuple[RunToolCatalogEntry, ...]
    described_deferred_references: tuple[DeferredToolReference, ...]
    described_schema_bytes: int


@dataclass(frozen=True, slots=True)
class _ResolvedCall:
    outer_call: ToolCall
    target_call: ToolCall
    entry: RunToolCatalogEntry | None = None
    control_name: str | None = None
    failure: ToolResultBlock | None = None
    validated_arguments: FrozenJsonObject | None = None
    control_result: ToolResultBlock | None = None


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


_CONTROL_TOOL_NAMES = frozenset({"tool_search", "tool_describe", "tool_call"})
_TOKEN = re.compile(r"[a-z0-9]+")


def _control_definitions(limits: LoopLimits) -> tuple[ToolDefinition, ...]:
    return (
        ToolDefinition(
            name="tool_search",
            description=(
                "Search trusted metadata for applicable direct and deferred tools. "
                "Search does not grant execution authority."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": limits.max_tool_search_query_characters,
                    },
                    "domains": {
                        "type": "array",
                        "description": "Optional owner filter; omit when uncertain.",
                        "items": {"type": "string", "minLength": 1, "maxLength": 128},
                        "maxItems": limits.max_domain_manifest_entries,
                        "uniqueItems": True,
                    },
                    "data_access": {
                        "type": "string",
                        "description": (
                            "External/source-data filter; lifecycle reads use none. "
                            "Omit when uncertain."
                        ),
                        "enum": [item.value for item in AccessMode],
                    },
                    "operational_effect": {
                        "type": "string",
                        "enum": [item.value for item in OperationalEffect],
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": limits.max_tool_search_results,
                        "default": min(5, limits.max_tool_search_results),
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="tool_describe",
            description=(
                "Describe one exact applicable tool. Deferred descriptions return a "
                "run-bound reference for a later tool_call step."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tool_name": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    }
                },
                "required": ["tool_name"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="tool_call",
            description=(
                "Invoke one deferred tool using an exact reference returned by "
                "tool_describe on an earlier completed step."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tool_ref": {
                        "type": "string",
                        "pattern": r"^toolref:sha256:[0-9a-f]{64}$",
                    },
                    "arguments": {"type": "object"},
                },
                "required": ["tool_ref", "arguments"],
                "additionalProperties": False,
            },
        ),
    )


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

    async def execute_internal(
        self,
        request: InternalCapabilityRequest,
    ) -> InternalCapabilityOutcome:
        """Execute one exact internal-only capability through ordinary owners."""

        if not isinstance(request, InternalCapabilityRequest):
            raise TypeError("request must be InternalCapabilityRequest")
        capability, executor, owner_id = self._registry.resolve_internal_execution(
            request.capability_id,
            request.contract_digest,
        )
        if capability.operational_effect is not OperationalEffect.NONE:
            raise ValueError(
                "Phase B internal execution cannot bypass operational-effect governance"
            )
        domain = self._domains[owner_id]
        call = ToolCall(
            id=request.call_id,
            name=f"internal:{capability.id}",
            arguments=request.arguments,
        )
        started = (
            asyncio.get_running_loop().time() if self._observer is not None else None
        )
        self._emit_tool_started(
            request.run,
            call,
            capability,
            invocation_mode=None,
        )
        try:
            normalized = domain.normalize_arguments(capability, request.arguments)
            arguments = self._registry.validate_arguments(capability.id, normalized)
            arguments = await domain.prepare_call(
                request.run,
                call,
                capability,
                arguments,
                request_sensitivity=request.sensitivity,
            )
            current, current_executor, current_owner = (
                self._registry.resolve_internal_execution(
                    request.capability_id,
                    request.contract_digest,
                )
            )
            if (
                current != capability
                or current_executor is not executor
                or current_owner != owner_id
            ):
                raise ValueError("internal execution identity changed")
            output, artifact_ref = await self._execute_no_effect(
                request.run,
                call,
                capability,
                executor,
                arguments,
                domain,
                sensitivity=request.sensitivity,
                reserved_artifact_id=request.reserved_artifact_id,
            )
            result = _bounded_tool_result(
                call,
                _classified_success(call, output, artifact_ref=artifact_ref),
                self._limits,
            )
            if result.is_error:
                raise ToolOutputValidationError(
                    "internal capability output exceeded the ordinary runtime result contract"
                )
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            result = _bounded_tool_result(
                call,
                self._exception_result(call, error, domain),
                self._limits,
            )
            self._emit_tool_completed(
                request.run,
                call,
                result,
                started,
                capability=capability,
                invocation_mode=None,
            )
            raise
        self._emit_tool_completed(
            request.run,
            call,
            result,
            started,
            capability=capability,
            invocation_mode=None,
        )
        return InternalCapabilityOutcome(output=output, artifact_ref=artifact_ref)

    async def prepare_run(self, run: RunInput) -> RunToolCatalog:
        """Prepare the complete immutable applicable catalog exactly once."""

        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        projected: dict[str, RunToolCatalogEntry] = {}
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
                schema_digest = _sha256_digest(capability.input_schema)
                origin_digest = view.origin_revision_digest or _sha256_digest(
                    {
                        "domain_owner_id": owner_id,
                        "capability_id": capability.id,
                        "executor_id": capability.executor_id,
                        "input_schema_digest": schema_digest,
                    }
                )
                projected[name] = RunToolCatalogEntry(
                    view=view,
                    capability=capability,
                    domain_owner_id=owner_id,
                    executor_id=capability.executor_id,
                    input_schema_digest=schema_digest,
                    origin_revision_digest=origin_digest,
                    invocation_mode=ToolInvocationMode.DIRECT,
                )
        entries = _select_invocation_modes(
            tuple(projected[name] for name in sorted(projected)),
            self._registry,
            self._limits,
        )
        manifest = _domain_manifest(entries)
        manifest_bytes = len(canonical_json(_manifest_material(manifest)).encode())
        if (
            len(manifest) > self._limits.max_domain_manifest_entries
            or manifest_bytes > self._limits.max_domain_manifest_bytes
            or (manifest_bytes + 3) // 4 > self._limits.max_domain_manifest_tokens
        ):
            raise ToolManifestLimitExceeded()
        execution_scope_digest = _sha256_digest(
            {
                "run_id": run.id,
                "agent_id": run.agent_id,
                "source_id": run.source_id,
                "conversation_source_id": run.conversation_source_id,
            }
        )
        catalog_material = {
            "run_id": run.id,
            "agent_id": run.agent_id,
            "execution_scope_digest": execution_scope_digest,
            "registry_digest": self._registry.digest,
            "entries": [_catalog_entry_material(entry) for entry in entries],
            "domain_manifest": _manifest_material(manifest),
        }
        aggregate_bytes = len(canonical_json(catalog_material).encode("utf-8"))
        if (
            len(entries) > self._limits.max_run_tool_catalog_entries
            or aggregate_bytes > self._limits.max_run_tool_catalog_bytes
        ):
            raise ToolCatalogLimitExceeded(
                observed_tools=len(entries),
                maximum_tools=self._limits.max_run_tool_catalog_entries,
                observed_catalog_bytes=aggregate_bytes,
                maximum_catalog_bytes=self._limits.max_run_tool_catalog_bytes,
            )
        catalog_digest = _sha256_digest(catalog_material)
        definitions = tuple(
            sorted(
                (
                    *(
                        self._registry.tool_definition(entry.view.name)
                        for entry in entries
                        if entry.invocation_mode is ToolInvocationMode.DIRECT
                    ),
                    *(
                        _control_definitions(self._limits)
                        if any(
                            entry.invocation_mode is ToolInvocationMode.DEFERRED
                            for entry in entries
                        )
                        else ()
                    ),
                ),
                key=lambda item: item.name,
            )
        )
        _validate_direct_surface(definitions, self._limits)
        return RunToolCatalog(
            run_id=run.id,
            agent_id=run.agent_id,
            execution_scope_digest=execution_scope_digest,
            registry_digest=self._registry.digest,
            catalog_digest=catalog_digest,
            entries=entries,
            domain_manifest=manifest,
            provider_definitions=definitions,
            aggregate_bytes=aggregate_bytes,
            manifest_bytes=manifest_bytes,
            manifest_token_limit=self._limits.max_domain_manifest_tokens,
        )

    def project(
        self,
        catalog: object,
        messages: tuple[object, ...],
    ) -> StepToolProjection:
        """Derive one exact step projection from a catalog and current transcript."""

        if not isinstance(catalog, RunToolCatalog):
            raise TypeError("catalog must be RunToolCatalog")
        references, described_bytes = _descriptor_receipts(
            catalog,
            messages,
            self._limits,
        )
        direct = tuple(
            entry
            for entry in catalog.entries
            if entry.invocation_mode is ToolInvocationMode.DIRECT
        )
        projection_material = {
            "run_id": catalog.run_id,
            "catalog_digest": catalog.catalog_digest,
            "provider_definitions": [
                _definition_material(item) for item in catalog.provider_definitions
            ],
            "direct_tools": [entry.view.name for entry in direct],
            "described_references": [item.tool_ref for item in references],
        }
        return StepToolProjection(
            run_id=catalog.run_id,
            catalog_digest=catalog.catalog_digest,
            projection_digest=_sha256_digest(projection_material),
            provider_definitions=catalog.provider_definitions,
            catalog_entries=catalog.entries,
            direct_resolution_entries=direct,
            described_deferred_references=references,
            described_schema_bytes=described_bytes,
        )

    async def execute_all(
        self,
        run: RunInput,
        calls: tuple[ToolCall, ...],
        *,
        projection: object,
        sensitivity: ModelSensitivity,
    ) -> ToolBatchOutcome:
        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        if not isinstance(sensitivity, ModelSensitivity):
            raise TypeError("tool batch sensitivity must be ModelSensitivity")
        if not isinstance(projection, StepToolProjection):
            raise TypeError("projection must be StepToolProjection")
        if projection.run_id != run.id:
            raise ValueError("step projection belongs to another run")
        calls = tuple(calls)
        if any(not isinstance(call, ToolCall) for call in calls):
            raise TypeError("calls must contain ToolCall records")
        if len(calls) > self._limits.max_tool_calls_per_response:
            raise ValueError("tool batch exceeds max_tool_calls_per_response")
        resolved_calls = self._resolve_calls(
            calls,
            projection,
            sensitivity=sensitivity,
        )
        results: list[ToolResultBlock | None] = [None] * len(calls)
        started = [False] * len(calls)
        reads: list[tuple[int, _ResolvedCall]] = []
        read_gate = asyncio.Semaphore(self._limits.max_parallel_reads)
        source_gates: dict[str, asyncio.Semaphore] = {}

        async def execute_read(index: int, resolved: _ResolvedCall) -> ToolResultBlock:
            source_key = _source_pressure_key(run, resolved.target_call)
            source_gate = source_gates.setdefault(
                source_key,
                asyncio.Semaphore(self._limits.max_parallel_reads_per_source),
            )
            async with read_gate:
                async with source_gate:
                    started[index] = True
                    return await self._execute_resolved(
                        run,
                        resolved,
                        projection,
                        sensitivity=sensitivity,
                    )

        async def finish_reads() -> ToolBatchInterruption | None:
            if not reads:
                return None
            indexes = tuple(index for index, _ in reads)
            read_calls = tuple(resolved.outer_call for _, resolved in reads)
            tasks = tuple(
                asyncio.create_task(execute_read(index, resolved))
                for index, resolved in reads
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

        for index, resolved in enumerate(resolved_calls):
            call = resolved.outer_call
            if self._is_effectful(resolved):
                read_interruption = await finish_reads()
                if read_interruption is not None:
                    return _interrupted_batch(
                        resolved_calls,
                        results,
                        started,
                        read_interruption,
                        ToolBatchCertainty.DEFINITE,
                        self._limits,
                    )
                started[index] = True
                try:
                    results[index] = await self._execute_resolved(
                        run,
                        resolved,
                        projection,
                        sensitivity=sensitivity,
                    )
                except _ToolExecutionInterrupted as interrupted:
                    results[index] = interrupted.result
                    return _interrupted_batch(
                        resolved_calls,
                        results,
                        started,
                        interrupted.kind,
                        interrupted.certainty,
                        self._limits,
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
                        resolved_calls,
                        results,
                        started,
                        interruption,
                        ToolBatchCertainty.DEFINITE,
                        self._limits,
                    )
            else:
                reads.append((index, resolved))
        read_interruption = await finish_reads()
        if read_interruption is not None:
            return _interrupted_batch(
                resolved_calls,
                results,
                started,
                read_interruption,
                ToolBatchCertainty.DEFINITE,
                self._limits,
            )
        if any(result is None for result in results):
            raise RuntimeError("tool result scheduling left an incomplete call")
        ordered = cast(tuple[ToolResultBlock, ...], tuple(results))
        return ToolBatchOutcome(
            tuple(
                _bounded_tool_result(
                    resolved.outer_call,
                    _with_invocation_audit(result, resolved),
                    self._limits,
                )
                for resolved, result in zip(resolved_calls, ordered, strict=True)
            )
        )

    def _resolve_calls(
        self,
        calls: tuple[ToolCall, ...],
        projection: StepToolProjection,
        *,
        sensitivity: ModelSensitivity,
    ) -> tuple[_ResolvedCall, ...]:
        described_bytes = projection.described_schema_bytes
        references = {
            item.tool_ref for item in projection.described_deferred_references
        }
        resolved_calls: list[_ResolvedCall] = []
        for call in calls:
            resolved = self._resolve_call(call, projection)
            if resolved.control_name == "tool_describe":
                result, admitted_bytes, reference = self._tool_describe(
                    resolved.target_call,
                    projection,
                    sensitivity=sensitivity,
                    described_bytes=described_bytes,
                    described_references=frozenset(references),
                )
                resolved = replace(resolved, control_result=result)
                if not result.is_error:
                    described_bytes += admitted_bytes
                    if reference is not None:
                        references.add(reference)
            resolved_calls.append(resolved)
        return tuple(resolved_calls)

    def _resolve_call(
        self,
        call: ToolCall,
        projection: StepToolProjection,
    ) -> _ResolvedCall:
        direct = {
            entry.view.name: entry for entry in projection.direct_resolution_entries
        }
        entry = direct.get(call.name)
        if entry is not None:
            return _ResolvedCall(call, call, entry=entry)
        control_names = {item.name for item in projection.provider_definitions}
        if call.name not in _CONTROL_TOOL_NAMES or call.name not in control_names:
            return _ResolvedCall(
                call,
                call,
                failure=_error(
                    call,
                    "tool_not_available",
                    "The requested tool is not available for this step projection.",
                    {"tool_name": call.name},
                ),
            )
        definition = next(
            item for item in projection.provider_definitions if item.name == call.name
        )
        try:
            arguments = validate_tool_schema_value(
                definition.input_schema,
                call.arguments,
            )
        except (TypeError, ValueError, RuntimeError):
            code = (
                "tool_reference_invalid"
                if call.name == "tool_call"
                else "invalid_tool_control_arguments"
            )
            return _ResolvedCall(
                call,
                call,
                failure=_error(
                    call,
                    code,
                    "The discovery control arguments are invalid.",
                ),
            )
        if call.name != "tool_call":
            return _ResolvedCall(
                call, replace(call, arguments=arguments), control_name=call.name
            )
        reference_value = arguments.get("tool_ref")
        nested = arguments.get("arguments")
        reference = next(
            (
                item
                for item in projection.described_deferred_references
                if item.tool_ref == reference_value
            ),
            None,
        )
        if reference is None or not isinstance(nested, Mapping):
            return _ResolvedCall(
                call,
                call,
                failure=_error(
                    call,
                    "tool_reference_invalid",
                    "The deferred tool reference is not valid for this run and step.",
                ),
            )
        view: ToolView | None
        capability: Capability | None
        owner_id: str | None
        try:
            resolved_view, resolved_capability, resolved_owner_id = (
                self._registry.resolve_tool_owner(reference.tool_name)
            )
            view = resolved_view
            capability = resolved_capability
            owner_id = resolved_owner_id
        except KeyError:
            view = capability = owner_id = None
        if (
            view is None
            or capability is None
            or owner_id is None
            or capability.id != reference.capability_id
            or capability.executor_id != reference.executor_id
            or owner_id != reference.domain_owner_id
            or _sha256_digest(capability.input_schema) != reference.input_schema_digest
        ):
            return _ResolvedCall(
                call,
                call,
                failure=_error(
                    call,
                    "tool_reference_invalid",
                    "The deferred tool reference no longer resolves exactly.",
                ),
            )
        deferred_entry = next(
            (
                item
                for item in projection.catalog_entries
                if item.view.name == reference.tool_name
                and item.invocation_mode is ToolInvocationMode.DEFERRED
                and item.view == view
                and item.capability == capability
                and item.domain_owner_id == reference.domain_owner_id
                and item.executor_id == reference.executor_id
                and item.input_schema_digest == reference.input_schema_digest
                and item.origin_revision_digest == reference.origin_revision_digest
            ),
            None,
        )
        if deferred_entry is None:
            return _ResolvedCall(
                call,
                call,
                failure=_error(
                    call,
                    "tool_reference_invalid",
                    "The deferred tool reference is not an exact catalog entry.",
                ),
            )
        target = ToolCall(id=call.id, name=view.name, arguments=nested)
        domain = self._domains[owner_id]
        try:
            normalized = domain.normalize_arguments(
                capability,
                nested,
            )
            validated = self._registry.validate_arguments(
                capability.id,
                normalized,
            )
        except Exception as error:
            return _ResolvedCall(
                call,
                target,
                entry=deferred_entry,
                failure=self._exception_result(target, error, domain),
            )
        target = replace(target, arguments=validated)
        return _ResolvedCall(
            call,
            target,
            entry=deferred_entry,
            validated_arguments=validated,
        )

    async def _execute_resolved(
        self,
        run: RunInput,
        resolved: _ResolvedCall,
        projection: StepToolProjection,
        *,
        sensitivity: ModelSensitivity,
    ) -> ToolResultBlock:
        if resolved.failure is not None:
            call = (
                resolved.target_call
                if resolved.entry is not None
                else resolved.outer_call
            )
            capability = (
                resolved.entry.capability if resolved.entry is not None else None
            )
            invocation_mode = (
                resolved.entry.invocation_mode if resolved.entry is not None else None
            )
            started = (
                asyncio.get_running_loop().time()
                if self._observer is not None
                else None
            )
            self._emit_tool_started(
                run,
                call,
                capability,
                invocation_mode=invocation_mode,
            )
            result = _bounded_tool_result(
                resolved.outer_call,
                resolved.failure,
                self._limits,
            )
            self._emit_tool_completed(
                run,
                call,
                result,
                started,
                capability=capability,
                invocation_mode=invocation_mode,
            )
            return result
        if resolved.control_name is not None:
            return await self._execute_control(
                run,
                resolved.target_call,
                resolved.control_name,
                projection,
                sensitivity=sensitivity,
                prepared_result=resolved.control_result,
            )
        if resolved.entry is None:
            raise RuntimeError("resolved capability call omitted its catalog entry")
        return await self._execute_one(
            run,
            resolved.target_call,
            resolved.entry,
            sensitivity=sensitivity,
            validated_arguments=resolved.validated_arguments,
        )

    async def _execute_control(
        self,
        run: RunInput,
        call: ToolCall,
        control_name: str,
        projection: StepToolProjection,
        *,
        sensitivity: ModelSensitivity,
        prepared_result: ToolResultBlock | None,
    ) -> ToolResultBlock:
        started = (
            asyncio.get_running_loop().time() if self._observer is not None else None
        )
        self._emit_tool_started(run, call, None, invocation_mode=None)
        if prepared_result is not None:
            result = prepared_result
        elif control_name == "tool_search":
            result = self._tool_search(call, projection, sensitivity=sensitivity)
        else:
            result = _error(
                call,
                "tool_not_available",
                "The requested discovery control is unavailable.",
            )
        result = _bounded_tool_result(call, result, self._limits)
        self._emit_tool_completed(
            run,
            call,
            result,
            started,
            capability=None,
            invocation_mode=None,
        )
        return result

    def _tool_search(
        self,
        call: ToolCall,
        projection: StepToolProjection,
        *,
        sensitivity: ModelSensitivity,
    ) -> ToolResultBlock:
        query = call.arguments.get("query")
        domains_value = call.arguments.get("domains", ())
        access_value = call.arguments.get("data_access")
        operational_effect = call.arguments.get("operational_effect")
        limit_value = call.arguments.get(
            "limit", min(5, self._limits.max_tool_search_results)
        )
        if (
            not isinstance(query, str)
            or not isinstance(domains_value, (tuple, list))
            or not isinstance(limit_value, int)
            or isinstance(limit_value, bool)
        ):
            return _error(
                call,
                "invalid_tool_control_arguments",
                "The tool search arguments are invalid.",
            )
        domains = frozenset(item for item in domains_value if isinstance(item, str))
        scored: list[tuple[int, str, str, RunToolCatalogEntry]] = []
        for entry in projection.catalog_entries:
            if domains and entry.domain_owner_id not in domains:
                continue
            if (
                access_value is not None
                and entry.capability.access_mode.value != access_value
            ):
                continue
            if (
                operational_effect is not None
                and entry.capability.operational_effect.value != operational_effect
            ):
                continue
            score = _tool_search_score(query, entry)
            if score > 0:
                scored.append((score, entry.domain_owner_id, entry.view.name, entry))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        total = len(scored)
        selected = scored[:limit_value]
        matches = [_tool_search_match(score, entry) for score, _, _, entry in selected]
        while True:
            data = {
                "catalog_digest": projection.catalog_digest,
                "matches": matches,
                "total_matches": total,
                "returned_count": len(matches),
                "truncated": len(matches) < total,
            }
            if (
                len(canonical_json(data).encode("utf-8"))
                <= self._limits.max_tool_search_result_bytes
            ):
                return _control_success(
                    call,
                    "tool_search_result",
                    data,
                    sensitivity=sensitivity,
                    run_id=projection.run_id,
                    catalog_digest=projection.catalog_digest,
                )
            if not matches:
                return _error(
                    call,
                    "tool_search_limit_exceeded",
                    "The bounded tool search result cannot fit its byte limit.",
                )
            matches.pop()

    def _tool_describe(
        self,
        call: ToolCall,
        projection: StepToolProjection,
        *,
        sensitivity: ModelSensitivity,
        described_bytes: int,
        described_references: frozenset[str],
    ) -> tuple[ToolResultBlock, int, str | None]:
        tool_name = call.arguments.get("tool_name")
        entry = next(
            (
                item
                for item in projection.catalog_entries
                if item.view.name == tool_name
            ),
            None,
        )
        if entry is None:
            return (
                _error(
                    call,
                    "tool_not_available",
                    "The requested tool is not in this run catalog.",
                    {"tool_name": tool_name if isinstance(tool_name, str) else ""},
                ),
                0,
                None,
            )
        data: dict[str, object] = {
            "tool_name": entry.view.name,
            "capability_id": entry.capability.id,
            "domain_owner_id": entry.domain_owner_id,
            "executor_id": entry.executor_id,
            "description": entry.view.description,
            "summary": entry.view.discovery.summary,
            "when_to_use": entry.view.discovery.when_to_use,
            "keywords": entry.view.discovery.keywords,
            "input_schema": entry.capability.input_schema,
            "output_kind": entry.capability.output_kind,
            "data_access": entry.capability.access_mode.value,
            "operational_effect": entry.capability.operational_effect.value,
            "invocation_mode": entry.invocation_mode.value,
            "input_schema_digest": entry.input_schema_digest,
            "origin_revision_digest": entry.origin_revision_digest,
            "catalog_digest": projection.catalog_digest,
        }
        reference: str | None = None
        if entry.invocation_mode is ToolInvocationMode.DEFERRED:
            reference = _tool_reference(
                projection.run_id,
                projection.catalog_digest,
                entry,
            )
            data["tool_ref"] = reference
        description_bytes = len(canonical_json(data).encode("utf-8"))
        if (
            description_bytes > self._limits.max_tool_description_bytes
            or described_bytes + description_bytes
            > self._limits.max_tool_description_bytes_per_run
            or (
                entry.invocation_mode is ToolInvocationMode.DEFERRED
                and reference not in described_references
                and len(described_references)
                >= self._limits.max_tool_references_per_run
            )
        ):
            return (
                _error(
                    call,
                    "tool_description_limit_exceeded",
                    "The exact tool description exceeds its individual or cumulative bound.",
                ),
                0,
                None,
            )
        data["description_bytes"] = description_bytes
        return (
            _control_success(
                call,
                "tool_description",
                data,
                sensitivity=sensitivity,
                run_id=projection.run_id,
                catalog_digest=projection.catalog_digest,
            ),
            description_bytes,
            reference,
        )

    async def _execute_one(
        self,
        run: RunInput,
        call: ToolCall,
        entry: RunToolCatalogEntry,
        *,
        sensitivity: ModelSensitivity,
        validated_arguments: FrozenJsonObject | None,
    ) -> ToolResultBlock:
        started = (
            asyncio.get_running_loop().time() if self._observer is not None else None
        )
        capability = entry.capability
        self._emit_tool_started(
            run,
            call,
            capability,
            invocation_mode=entry.invocation_mode,
        )
        interruption_kind: ToolBatchInterruption | None = None
        outcome_certainty = ToolBatchCertainty.DEFINITE
        domain: CapabilityDomain | None = None
        try:
            view, resolved, owner_id = self._registry.resolve_tool_owner(call.name)
            if (
                view != entry.view
                or resolved != entry.capability
                or owner_id != entry.domain_owner_id
                or resolved.executor_id != entry.executor_id
                or _sha256_digest(resolved.input_schema) != entry.input_schema_digest
            ):
                raise ValueError("tool catalog execution identity changed")
            capability = resolved
            domain = self._domains[owner_id]
            if validated_arguments is None:
                raw_arguments = domain.normalize_arguments(capability, call.arguments)
                arguments = self._registry.validate_arguments(
                    capability.id,
                    raw_arguments,
                )
            else:
                arguments = validated_arguments
            arguments = await domain.prepare_call(
                run,
                call,
                capability,
                arguments,
                request_sensitivity=sensitivity,
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
            if capability.operational_effect is not OperationalEffect.NONE:
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
                    sensitivity=sensitivity,
                    invocation_mode=entry.invocation_mode,
                )
            else:
                output, artifact_ref = await self._execute_no_effect(
                    run,
                    call,
                    capability,
                    executor,
                    arguments,
                    domain,
                    sensitivity=sensitivity,
                )
                result = _classified_success(call, output, artifact_ref=artifact_ref)
        except _ToolExecutionInterrupted:
            raise
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            result = self._exception_result(call, error, domain)
        result = _bounded_tool_result(call, result, self._limits)
        self._emit_tool_completed(
            run,
            call,
            result,
            started,
            capability=capability,
            invocation_mode=entry.invocation_mode,
        )
        if interruption_kind is not None:
            raise _ToolExecutionInterrupted(
                result,
                interruption_kind,
                outcome_certainty,
            )
        return result

    async def _execute_no_effect(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        executor: Executor,
        arguments: FrozenJsonObject,
        domain: CapabilityDomain,
        *,
        sensitivity: ModelSensitivity,
        reserved_artifact_id: str | None = None,
    ) -> tuple[ToolOutput, ArtifactRef | None]:
        if capability.operational_effect is not OperationalEffect.NONE:
            raise ValueError("non-effect execution requires operational effect none")
        execution = ToolExecution(
            run_id=run.id,
            call_id=call.id,
            capability_id=capability.id,
            arguments=arguments,
            conversation_id=run.conversation_id or run.id,
        )
        candidate = await executor.execute(execution)
        if not isinstance(candidate, ToolOutput):
            raise ToolOutputValidationError("executor did not return ToolOutput")
        output = await domain.finalize_output(
            run,
            call,
            capability,
            arguments,
            candidate,
            request_sensitivity=sensitivity,
        )
        output = self._registry.validate_output(capability.id, output)
        artifact_ref = await self._commit_artifact_output(
            run,
            call,
            capability,
            output,
            reserved_artifact_id=reserved_artifact_id,
        )
        return output, artifact_ref

    async def _execute_side_effect(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        executor: Executor,
        execution: ToolExecution,
        arguments: FrozenJsonObject,
        domain: CapabilityDomain,
        *,
        sensitivity: ModelSensitivity,
        invocation_mode: ToolInvocationMode,
    ) -> tuple[
        ToolResultBlock,
        ToolBatchInterruption | None,
        ToolBatchCertainty,
    ]:
        if capability.operational_effect is OperationalEffect.NONE:
            raise ValueError("effect execution requires an operational effect")
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
            self._emit_approval_requested(
                run,
                call,
                capability,
                invocation_mode=invocation_mode,
            )
            try:
                decision = await self._approval_handler(request)
            except asyncio.CancelledError:
                raise
            except Exception:
                self._emit_approval_decided(
                    run,
                    call,
                    capability,
                    "failed",
                    invocation_mode=invocation_mode,
                )
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
                self._emit_approval_decided(
                    run,
                    call,
                    capability,
                    "failed",
                    invocation_mode=invocation_mode,
                )
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
                self._emit_approval_decided(
                    run,
                    call,
                    capability,
                    "denied",
                    invocation_mode=invocation_mode,
                )
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
            self._emit_approval_decided(
                run,
                call,
                capability,
                "approved",
                invocation_mode=invocation_mode,
            )
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
            sensitivity=sensitivity,
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
        *,
        sensitivity: ModelSensitivity,
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
                request_sensitivity=sensitivity,
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
        *,
        reserved_artifact_id: str | None = None,
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
            reserved_artifact_id=reserved_artifact_id,
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

    def _is_effectful(
        self,
        resolved: _ResolvedCall,
    ) -> bool:
        return (
            resolved.failure is None
            and resolved.entry is not None
            and resolved.entry.capability.operational_effect
            is not OperationalEffect.NONE
        )

    def _emit_tool_started(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability | None,
        *,
        invocation_mode: ToolInvocationMode | None,
    ) -> None:
        data: dict[str, object] = {"call_id": call.id, "tool_name": call.name}
        if capability is not None:
            data["capability_id"] = capability.id
        if invocation_mode is not None:
            data["invocation_mode"] = invocation_mode.value
        self._emit(AgentEventKind.TOOL_STARTED, run, data)

    def _emit_approval_requested(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        *,
        invocation_mode: ToolInvocationMode,
    ) -> None:
        self._emit(
            AgentEventKind.APPROVAL_REQUESTED,
            run,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "capability_id": capability.id,
                "invocation_mode": invocation_mode.value,
            },
        )

    def _emit_approval_decided(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        outcome: str,
        *,
        invocation_mode: ToolInvocationMode,
    ) -> None:
        self._emit(
            AgentEventKind.APPROVAL_DECIDED,
            run,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "capability_id": capability.id,
                "invocation_mode": invocation_mode.value,
                "outcome": outcome,
            },
        )

    def _emit_tool_completed(
        self,
        run: RunInput,
        call: ToolCall,
        result: ToolResultBlock,
        started: float | None,
        *,
        capability: Capability | None,
        invocation_mode: ToolInvocationMode | None,
    ) -> None:
        if self._observer is None:
            return
        assert started is not None
        data: dict[str, object] = {
            "call_id": call.id,
            "tool_name": call.name,
            "duration_ms": _duration_ms(started),
            "success": not result.is_error,
            "error_code": _result_error_code(result),
        }
        if capability is not None:
            data["capability_id"] = capability.id
        if invocation_mode is not None:
            data["invocation_mode"] = invocation_mode.value
        self._emit(
            AgentEventKind.TOOL_COMPLETED,
            run,
            data,
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


def _sha256_digest(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _definition_material(definition: ToolDefinition) -> dict[str, object]:
    return {
        "name": definition.name,
        "description": definition.description,
        "input_schema": definition.input_schema,
    }


def _definition_bytes(definitions: tuple[ToolDefinition, ...]) -> int:
    return len(
        canonical_json([_definition_material(item) for item in definitions]).encode(
            "utf-8"
        )
    )


def _validate_direct_surface(
    definitions: tuple[ToolDefinition, ...],
    limits: LoopLimits,
) -> None:
    definition_bytes = _definition_bytes(definitions)
    if (
        len(definitions) > limits.max_direct_tools
        or definition_bytes > limits.max_direct_tool_definition_bytes
    ):
        raise ToolSurfaceLimitExceeded(
            observed_tools=len(definitions),
            maximum_tools=limits.max_direct_tools,
            observed_definition_bytes=definition_bytes,
            maximum_definition_bytes=limits.max_direct_tool_definition_bytes,
        )


def _select_invocation_modes(
    entries: tuple[RunToolCatalogEntry, ...],
    registry: CapabilityRegistry,
    limits: LoopLimits,
) -> tuple[RunToolCatalogEntry, ...]:
    if limits.tool_projection_mode is ToolProjectionMode.EAGER:
        return tuple(
            replace(entry, invocation_mode=ToolInvocationMode.DIRECT)
            for entry in entries
        )
    selected = {
        entry.view.name
        for entry in entries
        if entry.view.discovery.exposure_class is ToolExposureClass.CORE
    }
    if limits.tool_projection_mode is ToolProjectionMode.AUTO:
        candidates = sorted(
            (
                entry
                for entry in entries
                if entry.view.discovery.exposure_class is ToolExposureClass.STANDARD
            ),
            key=lambda item: (
                -item.view.discovery.eager_priority,
                item.domain_owner_id,
                item.view.name,
            ),
        )
        for candidate in candidates:
            proposed = (*selected, candidate.view.name)
            definitions = tuple(
                registry.tool_definition(name) for name in sorted(proposed)
            )
            if (
                len(definitions) <= limits.max_eager_tools
                and _definition_bytes(definitions)
                <= limits.max_eager_tool_definition_bytes
            ):
                selected.add(candidate.view.name)
    return tuple(
        replace(
            entry,
            invocation_mode=(
                ToolInvocationMode.DIRECT
                if entry.view.name in selected
                else ToolInvocationMode.DEFERRED
            ),
        )
        for entry in entries
    )


def _catalog_entry_material(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "domain_owner_id": entry.domain_owner_id,
        "executor_id": entry.executor_id,
        "description": entry.view.description,
        "input_schema": entry.capability.input_schema,
        "input_schema_digest": entry.input_schema_digest,
        "output_kind": entry.capability.output_kind,
        "data_access": entry.capability.access_mode.value,
        "operational_effect": entry.capability.operational_effect.value,
        "discovery": {
            "summary": entry.view.discovery.summary,
            "when_to_use": entry.view.discovery.when_to_use,
            "keywords": entry.view.discovery.keywords,
            "exposure_class": entry.view.discovery.exposure_class.value,
            "eager_priority": entry.view.discovery.eager_priority,
        },
        "origin_revision_digest": entry.origin_revision_digest,
        "invocation_mode": entry.invocation_mode.value,
    }


def _domain_manifest(
    entries: tuple[RunToolCatalogEntry, ...],
) -> tuple[DomainToolManifestEntry, ...]:
    owners = sorted({entry.domain_owner_id for entry in entries})
    return tuple(
        DomainToolManifestEntry(
            domain_owner_id=owner,
            summary=f"Trusted {owner} capability counts.",
            direct_count=sum(
                entry.domain_owner_id == owner
                and entry.invocation_mode is ToolInvocationMode.DIRECT
                for entry in entries
            ),
            deferred_count=sum(
                entry.domain_owner_id == owner
                and entry.invocation_mode is ToolInvocationMode.DEFERRED
                for entry in entries
            ),
        )
        for owner in owners
    )


def _manifest_material(
    manifest: tuple[DomainToolManifestEntry, ...],
) -> list[dict[str, object]]:
    return [
        {
            "domain_owner_id": item.domain_owner_id,
            "summary": item.summary,
            "direct_count": item.direct_count,
            "deferred_count": item.deferred_count,
        }
        for item in manifest
    ]


def _tool_reference(
    run_id: str,
    catalog_digest: str,
    entry: RunToolCatalogEntry,
) -> str:
    digest = _sha256_digest(
        {
            "run_id": run_id,
            "catalog_digest": catalog_digest,
            "tool_name": entry.view.name,
            "capability_id": entry.capability.id,
            "domain_owner_id": entry.domain_owner_id,
            "executor_id": entry.executor_id,
            "input_schema_digest": entry.input_schema_digest,
            "origin_revision_digest": entry.origin_revision_digest,
            "invocation_mode": entry.invocation_mode.value,
        }
    )
    return "toolref:" + digest


def _descriptor_receipts(
    catalog: RunToolCatalog,
    messages: tuple[object, ...],
    limits: LoopLimits,
) -> tuple[tuple[DeferredToolReference, ...], int]:
    calls: dict[str, str] = {}
    seen_results: set[str] = set()
    references: dict[str, DeferredToolReference] = {}
    described_bytes = 0
    entries = {entry.view.name: entry for entry in catalog.entries}
    for message in messages:
        if not isinstance(message, CanonicalMessage):
            raise TypeError("step transcript must contain CanonicalMessage records")
        if message.role is MessageRole.ASSISTANT:
            for call in message.tool_calls:
                if call.name == "tool_describe":
                    tool_name = call.arguments.get("tool_name")
                    if isinstance(tool_name, str):
                        calls[call.id] = tool_name
            continue
        if message.role is not MessageRole.TOOL:
            continue
        for block in message.content:
            if not isinstance(block, ToolResultBlock) or block.call_id in seen_results:
                continue
            if block.call_id not in calls or block.is_error:
                continue
            if block.output.get("kind") != "tool_description":
                continue
            data = block.output.get("data")
            provenance = block.sensitivity_provenance
            if not isinstance(data, Mapping) or (
                provenance.get("authority") != "tool_catalog_control"
                or provenance.get("run_id") != catalog.run_id
                or provenance.get("catalog_digest") != catalog.catalog_digest
            ):
                continue
            entry = entries.get(calls[block.call_id])
            if entry is None or not _descriptor_matches(data, catalog, entry):
                continue
            declared_bytes = data.get("description_bytes")
            without_bytes = {
                key: value for key, value in data.items() if key != "description_bytes"
            }
            actual_bytes = len(canonical_json(without_bytes).encode("utf-8"))
            if declared_bytes != actual_bytes:
                continue
            seen_results.add(block.call_id)
            described_bytes += actual_bytes
            if described_bytes > limits.max_tool_description_bytes_per_run:
                raise ValueError("descriptor transcript exceeds its cumulative bound")
            if entry.invocation_mode is ToolInvocationMode.DIRECT:
                continue
            tool_ref = data.get("tool_ref")
            expected_ref = _tool_reference(
                catalog.run_id, catalog.catalog_digest, entry
            )
            if tool_ref != expected_ref:
                continue
            references[expected_ref] = DeferredToolReference(
                tool_ref=expected_ref,
                tool_name=entry.view.name,
                capability_id=entry.capability.id,
                domain_owner_id=entry.domain_owner_id,
                executor_id=entry.executor_id,
                input_schema_digest=entry.input_schema_digest,
                origin_revision_digest=entry.origin_revision_digest,
                catalog_digest=catalog.catalog_digest,
                description_bytes=actual_bytes,
            )
    if len(references) > limits.max_tool_references_per_run:
        raise ValueError("descriptor transcript exceeds its reference bound")
    return tuple(references[key] for key in sorted(references)), described_bytes


def _descriptor_matches(
    data: Mapping[str, object],
    catalog: RunToolCatalog,
    entry: RunToolCatalogEntry,
) -> bool:
    keywords = data.get("keywords")
    if not isinstance(keywords, (tuple, list)):
        return False
    return all(
        (
            data.get("tool_name") == entry.view.name,
            data.get("capability_id") == entry.capability.id,
            data.get("domain_owner_id") == entry.domain_owner_id,
            data.get("executor_id") == entry.executor_id,
            data.get("description") == entry.view.description,
            data.get("summary") == entry.view.discovery.summary,
            data.get("when_to_use") == entry.view.discovery.when_to_use,
            tuple(keywords) == entry.view.discovery.keywords,
            data.get("input_schema") == entry.capability.input_schema,
            data.get("output_kind") == entry.capability.output_kind,
            data.get("data_access") == entry.capability.access_mode.value,
            data.get("operational_effect") == entry.capability.operational_effect.value,
            data.get("invocation_mode") == entry.invocation_mode.value,
            data.get("input_schema_digest") == entry.input_schema_digest,
            data.get("origin_revision_digest") == entry.origin_revision_digest,
            data.get("catalog_digest") == catalog.catalog_digest,
        )
    )


def _tool_search_score(query: str, entry: RunToolCatalogEntry) -> int:
    normalized = query.strip().lower()
    terms = tuple(_TOKEN.findall(normalized))
    if not terms:
        return 0
    score = 0
    if normalized == entry.view.name:
        score += 10_000
    if normalized == entry.domain_owner_id:
        score += 5_000
    weighted_fields = (
        (entry.view.name, 100),
        (entry.domain_owner_id, 50),
        (entry.view.discovery.summary, 20),
        (entry.view.discovery.when_to_use, 10),
        (" ".join(entry.view.discovery.keywords), 30),
        (" ".join(entry.parameter_names), 15),
    )
    for text, weight in weighted_fields:
        tokens = set(_TOKEN.findall(text.lower()))
        score += weight * sum(term in tokens for term in terms)
    return score


def _tool_search_match(
    score: int,
    entry: RunToolCatalogEntry,
) -> dict[str, object]:
    return {
        "tool_name": entry.view.name,
        "domain_owner_id": entry.domain_owner_id,
        "summary": entry.view.discovery.summary,
        "invocation_mode": entry.invocation_mode.value,
        "data_access": entry.capability.access_mode.value,
        "operational_effect": entry.capability.operational_effect.value,
        "parameter_names": entry.parameter_names,
        "input_schema_digest": entry.input_schema_digest,
        "score": score,
    }


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
            _with_invocation_audit(result, resolved),
            limits,
        )
        for resolved, result in zip(resolved_calls, ordered, strict=True)
    )
    return ToolBatchOutcome(
        ordered_results=bounded,
        interruption_kind=interruption,
        outcome_certainty=certainty,
    )


def _with_invocation_audit(
    result: ToolResultBlock,
    resolved: _ResolvedCall,
) -> ToolResultBlock:
    entry = resolved.entry
    if entry is None or entry.invocation_mode is not ToolInvocationMode.DEFERRED:
        return result
    output = dict(result.output)
    output["invocation"] = {
        "authority": "capability_runtime",
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "invocation_mode": entry.invocation_mode.value,
    }
    return replace(result, output=output)


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
    "DeferredToolReference",
    "DomainToolManifestEntry",
    "InternalCapabilityOutcome",
    "InternalCapabilityRequest",
    "RunToolCatalog",
    "RunToolCatalogEntry",
    "SideEffectPlan",
    "StepToolProjection",
    "ToolInvocationMode",
]
