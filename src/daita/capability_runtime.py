"""Validate, govern, execute, and normalize capabilities across static domains."""

from __future__ import annotations

import asyncio
import hmac
import re
import secrets
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, cast

from ._json import FrozenJsonObject, canonical_json
from .artifacts.models import (
    ArtifactError,
    ArtifactRef,
    artifact_ref_to_mapping,
    canonical_artifact_filename,
)
from .artifacts.store import AgentHomeArtifactStore
from .capabilities import (
    RESERVED_TOOL_NAMES,
    TOOLBOX_DEFINITIONS,
    AccessMode,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    AutomationEligibility,
    AutomationScopeProposal,
    Capability,
    CapabilityDeclarations,
    CapabilityGrant,
    CapabilityInputError,
    CapabilityRegistry,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ExecutionContractReader,
    Executor,
    OperationalEffect,
    SideEffectExecutor,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
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
    RunOrigin,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolBatchOutcome,
)
from .observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely
from .scope import EffectiveSourceScope

if TYPE_CHECKING:
    from .storage.sqlite_records import EffectReceipt


class EffectReceiptStore(Protocol):
    async def require_effects_unblocked(
        self, agent_id: str, *, run_id: str | None = None, routine_id: str | None = None
    ) -> None: ...
    async def load_effect_receipt_for_call(
        self, agent_id: str, run_id: str, call_id: str
    ) -> EffectReceipt | None: ...
    async def load_effect_receipt_for_operation(
        self, agent_id: str, operation_key: str
    ) -> EffectReceipt | None: ...
    async def start_effect_receipt(
        self,
        receipt: EffectReceipt,
        *,
        grant: CapabilityGrant | None = None,
        max_receipts_per_run: int = 64,
    ) -> EffectReceipt: ...
    async def finish_effect_receipt(self, receipt: EffectReceipt) -> EffectReceipt: ...


@dataclass(frozen=True, slots=True)
class CapabilityFailure:
    """One bounded domain-owned failure ready for common result rendering."""

    code: str
    message: str
    details: Mapping[str, object] = field(default_factory=dict)
    effect_observation: EffectObservation | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("capability failure code must be non-empty text")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("capability failure message must be non-empty text")
        if self.effect_observation is not None and not isinstance(
            self.effect_observation, EffectObservation
        ):
            raise TypeError("capability failure observation must be code-owned")
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
    capability_grant_digest: str | None = None
    effect_intent: FrozenJsonObject | None = None

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
        if (
            self.capability_grant_digest is not None
            and re.fullmatch(r"sha256:[0-9a-f]{64}", self.capability_grant_digest)
            is None
        ):
            raise ValueError("side-effect grant digest is invalid")
        if self.effect_intent is not None:
            if not isinstance(self.effect_intent, FrozenJsonObject):
                raise TypeError(
                    "side-effect intent must be domain-normalized frozen JSON"
                )
            if len(canonical_json(self.effect_intent).encode("utf-8")) > 32 * 1024:
                raise ValueError("side-effect intent exceeds its byte bound")


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

    async def prepare_automation_grant(
        self,
        capability: Capability,
        constraints: FrozenJsonObject,
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> FrozenJsonObject: ...

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


@dataclass(frozen=True, slots=True)
class ToolboxManifestEntry:
    toolbox_id: ToolboxId
    label: str
    summary: str
    pinned_count: int
    on_demand_count: int
    access_modes: tuple[AccessMode, ...]
    operational_effects: tuple[OperationalEffect, ...]


@dataclass(frozen=True, slots=True)
class RunToolCatalogEntry:
    view: ToolView
    capability: Capability
    domain_owner_id: str
    executor_id: str
    input_schema_digest: str
    origin_revision_digest: str
    toolbox_id: ToolboxId
    load_mode: ToolLoadMode

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
    toolbox_manifest: tuple[ToolboxManifestEntry, ...]
    pinned_provider_definitions: tuple[ToolDefinition, ...]
    control_definitions: tuple[ToolDefinition, ...]
    aggregate_bytes: int
    manifest_bytes: int
    manifest_token_limit: int
    source_scope: EffectiveSourceScope | None = None

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(entry.capability.id for entry in self.entries)

    @property
    def initial_provider_definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(
            sorted(
                (*self.pinned_provider_definitions, *self.control_definitions),
                key=lambda item: item.name,
            )
        )

    @property
    def manifest_payload(self) -> tuple[FrozenJsonObject, ...]:
        return tuple(
            FrozenJsonObject.from_mapping(item)
            for item in _manifest_material(self.toolbox_manifest)
        )


@dataclass(frozen=True, slots=True)
class StepToolProjection:
    run_id: str
    registry_digest: str
    catalog_digest: str
    transcript_digest: str
    projection_digest: str
    activation_digest: str
    provider_definitions: tuple[ToolDefinition, ...]
    catalog_entries: tuple[RunToolCatalogEntry, ...]
    callable_entries: tuple[RunToolCatalogEntry, ...]
    loaded_entries: tuple[RunToolCatalogEntry, ...]
    loaded_definition_bytes: int
    source_scope: EffectiveSourceScope | None = None

    def require_current(
        self,
        *,
        run_id: str,
        registry_digest: str,
        catalog_digest: str,
        messages: tuple[object, ...],
    ) -> StepToolProjection:
        """Return this projection only when it matches the exact current step."""

        if (
            self.run_id != run_id
            or self.registry_digest != registry_digest
            or self.catalog_digest != catalog_digest
            or self.transcript_digest != _toolbox_transcript_digest(messages)
            or self.projection_digest != _step_projection_digest(self)
        ):
            raise ValueError(
                "step tool projection differs from the current toolbox transcript"
            )
        return self


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


_CONTROL_TOOL_NAMES = RESERVED_TOOL_NAMES
_TOKEN = re.compile(r"[a-z0-9]+")


def _control_definitions(
    limits: LoopLimits, entries: tuple[RunToolCatalogEntry, ...]
) -> tuple[ToolDefinition, ...]:
    definitions = (
        ToolDefinition(
            name="toolbox_search",
            description=(
                "Find applicable tools by describing the task in query. Returns "
                "bounded metadata; does not load tools or grant authority."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": limits.max_toolbox_search_query_characters,
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": limits.max_toolbox_search_results,
                        "default": min(5, limits.max_toolbox_search_results),
                    },
                    "cursor": {"type": "string", "minLength": 1, "maxLength": 80},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="toolbox_load",
            description=(
                "Atomically load an exact on-demand working set. A successful load "
                "replaces the prior set on the next model step; ordinary validation "
                "and governance still apply. Retains bounded contracts in the result; "
                "use toolbox_inspect for omitted contracts without changing this set."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tool_names": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1, "maxLength": 128},
                        "minItems": 1,
                        "maxItems": limits.max_loaded_tools,
                        "uniqueItems": True,
                    }
                },
                "required": ["tool_names"],
                "additionalProperties": False,
            },
        ),
        ToolDefinition(
            name="toolbox_inspect",
            description=(
                "Inspect an exact prepared tool contract without loading or executing it. "
                "The result retains schemas for later composition. For a partial result, "
                "use its exact contract_digest, child path or next_offset to retrieve more. "
                "Paths use JSON Pointer; inspection grants no authority."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "tool_name": {"type": "string", "minLength": 1, "maxLength": 128},
                    "contract_digest": {
                        "type": "string",
                        "pattern": r"^sha256:[0-9a-f]{64}$",
                    },
                    "path": {"type": "string", "maxLength": 2048},
                    "offset": {
                        "type": "integer",
                        "minimum": 0,
                        "maximum": 2 * 1024 * 1024,
                    },
                },
                "required": ["tool_name"],
                "additionalProperties": False,
            },
        ),
    )

    if any(entry.load_mode is ToolLoadMode.ON_DEMAND for entry in entries):
        return definitions
    return (
        tuple(tool for tool in definitions if tool.name == "toolbox_inspect")
        if entries
        else ()
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
        effect_receipts: EffectReceiptStore | None = None,
        execution_contract_reader: ExecutionContractReader | None = None,
        limits: LoopLimits = LoopLimits(),
        side_effect_recovery_timeout_seconds: float | None = None,
        source_scope_resolver: (
            Callable[[RunInput], Awaitable[EffectiveSourceScope]] | None
        ) = None,
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
        self._source_scope_resolver = source_scope_resolver
        self._registry = registry
        self._search_cursor_key = secrets.token_bytes(32)
        self._domains = owners
        self._approval_handler = approval_handler
        self._mutation_lock = mutation_lock or asyncio.Lock()
        self._observer = observer
        self._clock = clock or (lambda: datetime.now(UTC))
        self._artifacts = artifacts
        self._effect_receipts = effect_receipts
        self._execution_contract_reader = execution_contract_reader
        self._effect_receipts_unavailable = False
        self._limits = limits
        self._side_effect_recovery_timeout_seconds = float(recovery_timeout)

    async def _validate_execution_contracts(self, run: RunInput) -> None:
        scope = run.execution_scope
        if scope is None:
            return
        if self._execution_contract_reader is None:
            raise CapabilityInputError(
                "execution_contract_unavailable",
                "Machine execution requires its current contract reader.",
            )
        try:
            current = await self._execution_contract_reader(
                agent_id=run.agent_id,
                source_ids=scope.allowed_source_ids,
                resource_ids=scope.allowed_resource_ids,
                capability_ids=scope.allowed_capability_ids,
                connector_binding_ids=scope.allowed_connector_binding_ids,
                model_route_ids=scope.eligible_model_routes,
            )
            origins = {
                view.capability_id
                for name in self._registry.tool_names
                for view, _ in (self._registry.resolve_tool(name),)
                if view.origin_revision_digest is not None
                and view.capability_id in scope.allowed_capability_ids
            }
            current.validate_coverage(
                capability_ids=scope.allowed_capability_ids,
                resource_ids=scope.allowed_resource_ids,
                route_ids=scope.eligible_model_routes,
                mcp_capability_ids=origins,
            )
            if any(
                current.capability_contracts[capability_id]
                != self._registry.contract_digest(capability_id)
                for capability_id in scope.allowed_capability_ids
            ):
                raise ValueError(
                    "current capability contract differs from the registry"
                )
        except (KeyError, ValueError) as error:
            raise CapabilityInputError(
                "execution_contract_unavailable",
                "An exact machine execution contract is unavailable.",
            ) from error
        if current != scope.contract_bindings:
            raise CapabilityInputError(
                "execution_contract_changed",
                "An approved execution contract changed; foreground approval is required for a revision.",
            )

    async def prepare_automation_grant(
        self,
        capability_id: str,
        requested_constraints: Mapping[str, object],
        max_calls_per_occurrence: int,
        proposal: AutomationScopeProposal,
    ) -> CapabilityGrant:
        """Normalize one proposed grant through its current owner; never execute it."""

        if not isinstance(proposal, AutomationScopeProposal):
            raise TypeError("grant preparation requires an immutable scope proposal")
        if (
            type(max_calls_per_occurrence) is not int
            or not 1 <= max_calls_per_occurrence <= 256
        ):
            raise CapabilityInputError(
                "automation_grant_invalid",
                "The requested call ceiling is outside its bound.",
            )
        try:
            capability, _ = self._registry.resolve_execution(capability_id)
            owner_id = self._registry.resolve_domain_owner(capability_id)
        except KeyError:
            raise CapabilityInputError(
                "automation_grant_unsupported", "The exact capability is unavailable."
            ) from None
        policy = capability.automation_grant_policy
        if (
            capability.automation_eligibility
            is not AutomationEligibility.AUTOMATION_DIRECT
            or policy is None
            or capability.effect_receipt_policy is None
        ):
            raise CapabilityInputError(
                "automation_grant_unsupported",
                "This capability does not support unattended effects.",
            )
        if (
            capability_id not in proposal.allowed_capability_ids
            or capability.access_mode not in proposal.allowed_access_modes
            or capability.operational_effect not in proposal.allowed_operational_effects
            or proposal.expires_at <= self._clock()
        ):
            raise CapabilityInputError(
                "automation_grant_outside_scope",
                "The proposed ceilings do not admit this effect.",
            )
        requested = self._registry.validate_grant_constraints(
            capability_id, requested_constraints
        )
        normalized = await self._domains[owner_id].prepare_automation_grant(
            capability, requested, max_calls_per_occurrence, proposal
        )
        if not isinstance(normalized, FrozenJsonObject):
            raise TypeError("domain grant preparation must return frozen constraints")
        normalized = self._registry.validate_grant_constraints(
            capability_id, normalized
        )
        return CapabilityGrant(
            # Identical preparation must survive the normal approval recheck.
            # This value confers authority only inside its exact enclosing routine scope.
            grant_id="capability-grant:"
            + _sha256_digest(
                {
                    "agent_id": proposal.agent_id,
                    "principal_id": proposal.principal_id,
                    "capability_contract": self._registry.contract_digest(
                        capability_id
                    ),
                    "domain_owner_id": owner_id,
                    "constraints_kind": policy.constraints_kind,
                    "constraints": normalized,
                    "max_calls_per_occurrence": max_calls_per_occurrence,
                }
            ).removeprefix("sha256:"),
            domain_owner_id=owner_id,
            capability_id=capability_id,
            capability_contract_digest=self._registry.contract_digest(capability_id),
            constraints_kind=policy.constraints_kind,
            constraints=normalized,
            max_calls_per_occurrence=max_calls_per_occurrence,
        )

    async def execute_internal(
        self,
        request: InternalCapabilityRequest,
    ) -> InternalCapabilityOutcome:
        """Execute one exact internal-only capability through ordinary owners."""

        if not isinstance(request, InternalCapabilityRequest):
            raise TypeError("request must be InternalCapabilityRequest")
        if self._source_scope_resolver is not None:
            request = replace(
                request,
                run=replace(
                    request.run,
                    resolved_source_scope=await self._source_scope_resolver(
                        request.run
                    ),
                ),
            )
        capability, executor, owner_id = self._registry.resolve_internal_execution(
            request.capability_id,
            request.contract_digest,
        )
        if capability.operational_effect is not OperationalEffect.NONE:
            raise ValueError(
                "internal execution cannot bypass operational-effect governance"
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
            catalog_entry=None,
        )
        try:
            await self._validate_execution_contracts(request.run)
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
                catalog_entry=None,
            )
            raise
        self._emit_tool_completed(
            request.run,
            call,
            result,
            started,
            capability=capability,
            catalog_entry=None,
        )
        return InternalCapabilityOutcome(output=output, artifact_ref=artifact_ref)

    async def prepare_run(self, run: RunInput) -> RunToolCatalog:
        """Prepare the complete immutable applicable catalog exactly once."""

        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        await self._validate_execution_contracts(run)
        if self._source_scope_resolver is not None:
            run = replace(
                run, resolved_source_scope=await self._source_scope_resolver(run)
            )
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
                if run.execution_scope is not None and not run.execution_scope.allows(
                    capability
                ):
                    continue
                if (
                    run.origin is RunOrigin.SCHEDULED_ROUTINE
                    and capability.automation_eligibility
                    is not AutomationEligibility.AUTOMATION_DIRECT
                ):
                    continue
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
                    toolbox_id=view.presentation.toolbox_id,
                    load_mode=view.presentation.load_mode,
                )
        entries = tuple(projected[name] for name in sorted(projected))
        manifest = _toolbox_manifest(entries)
        manifest_bytes = len(canonical_json(_manifest_material(manifest)).encode())
        if (
            len(manifest) > self._limits.max_toolbox_manifest_entries
            or manifest_bytes > self._limits.max_toolbox_manifest_bytes
            or (manifest_bytes + 3) // 4 > self._limits.max_toolbox_manifest_tokens
        ):
            raise ToolManifestLimitExceeded()
        execution_scope_digest = (
            run.execution_scope.digest
            if run.execution_scope is not None
            else _sha256_digest(
                {
                    "run_id": run.id,
                    "agent_id": run.agent_id,
                    "source_scope_ids": run.source_scope_ids,
                    "source_scope": (
                        None
                        if run.resolved_source_scope is None
                        else run.resolved_source_scope.to_mapping()
                    ),
                }
            )
        )
        catalog_material = {
            "run_id": run.id,
            "agent_id": run.agent_id,
            "execution_scope_digest": execution_scope_digest,
            "registry_digest": self._registry.digest,
            "entries": [_catalog_entry_material(entry) for entry in entries],
            "toolbox_manifest": _manifest_material(manifest),
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
        pinned_entries = tuple(
            entry for entry in entries if entry.load_mode is ToolLoadMode.PINNED
        )
        pinned_definitions = tuple(
            sorted(
                (
                    self._registry.tool_definition(entry.view.name)
                    for entry in pinned_entries
                ),
                key=lambda item: item.name,
            )
        )
        _validate_pinned_surface(pinned_definitions, self._limits)
        controls = _control_definitions(self._limits, entries)
        _validate_step_surface(
            tuple(sorted((*pinned_definitions, *controls), key=lambda item: item.name)),
            self._limits,
        )
        return RunToolCatalog(
            run_id=run.id,
            agent_id=run.agent_id,
            execution_scope_digest=execution_scope_digest,
            registry_digest=self._registry.digest,
            catalog_digest=catalog_digest,
            entries=entries,
            toolbox_manifest=manifest,
            pinned_provider_definitions=pinned_definitions,
            control_definitions=controls,
            aggregate_bytes=aggregate_bytes,
            manifest_bytes=manifest_bytes,
            manifest_token_limit=self._limits.max_toolbox_manifest_tokens,
            source_scope=run.resolved_source_scope,
        )

    def project(
        self,
        catalog: object,
        messages: tuple[object, ...],
    ) -> StepToolProjection:
        """Derive one exact step projection from a catalog and current transcript."""

        if not isinstance(catalog, RunToolCatalog):
            raise TypeError("catalog must be RunToolCatalog")
        if catalog.registry_digest != self._registry.digest:
            raise ValueError("run tool catalog registry identity changed")
        loaded, activation_digest, transcript_digest = _loaded_tool_receipt(
            run_id=catalog.run_id,
            catalog_digest=catalog.catalog_digest,
            catalog_entries=catalog.entries,
            messages=messages,
            limits=self._limits,
            registry=self._registry,
        )
        pinned = tuple(
            entry for entry in catalog.entries if entry.load_mode is ToolLoadMode.PINNED
        )
        callable_entries = tuple(
            sorted((*pinned, *loaded), key=lambda item: item.view.name)
        )
        provider_definitions = tuple(
            sorted(
                (
                    *(
                        self._registry.tool_definition(entry.view.name)
                        for entry in callable_entries
                    ),
                    *catalog.control_definitions,
                ),
                key=lambda item: item.name,
            )
        )
        _validate_step_surface(provider_definitions, self._limits)
        loaded_definition_bytes = _definition_bytes(
            tuple(self._registry.tool_definition(entry.view.name) for entry in loaded)
        )
        return StepToolProjection(
            run_id=catalog.run_id,
            registry_digest=catalog.registry_digest,
            catalog_digest=catalog.catalog_digest,
            transcript_digest=transcript_digest,
            projection_digest=_projection_digest(
                run_id=catalog.run_id,
                registry_digest=catalog.registry_digest,
                catalog_digest=catalog.catalog_digest,
                transcript_digest=transcript_digest,
                provider_definitions=provider_definitions,
                callable_entries=callable_entries,
                loaded_entries=loaded,
                activation_digest=activation_digest,
                source_scope=catalog.source_scope,
            ),
            activation_digest=activation_digest,
            provider_definitions=provider_definitions,
            catalog_entries=catalog.entries,
            callable_entries=callable_entries,
            loaded_entries=loaded,
            loaded_definition_bytes=loaded_definition_bytes,
            source_scope=catalog.source_scope,
        )

    async def execute_all(
        self,
        run: RunInput,
        calls: tuple[ToolCall, ...],
        *,
        projection: object,
        messages: tuple[CanonicalMessage, ...],
        sensitivity: ModelSensitivity,
    ) -> ToolBatchOutcome:
        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        if not isinstance(sensitivity, ModelSensitivity):
            raise TypeError("tool batch sensitivity must be ModelSensitivity")
        if not isinstance(projection, StepToolProjection):
            raise TypeError("projection must be StepToolProjection")
        projection = projection.require_current(
            run_id=run.id,
            registry_digest=self._registry.digest,
            catalog_digest=projection.catalog_digest,
            messages=messages,
        )
        loaded, activation_digest, transcript_digest = _loaded_tool_receipt(
            run_id=projection.run_id,
            catalog_digest=projection.catalog_digest,
            catalog_entries=projection.catalog_entries,
            messages=messages,
            limits=self._limits,
            registry=self._registry,
        )
        if (
            projection.loaded_entries != loaded
            or projection.activation_digest != activation_digest
            or projection.transcript_digest != transcript_digest
        ):
            raise ValueError(
                "step projection differs from the current toolbox transcript"
            )
        _validate_step_projection(projection, self._registry, self._limits)
        run = replace(run, resolved_source_scope=projection.source_scope)
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
                    result,
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
        resolved_calls: list[_ResolvedCall] = []
        load_succeeded = False
        for call in calls:
            resolved = self._resolve_call(call, projection)
            if resolved.control_name == "toolbox_load":
                if load_succeeded:
                    result = _error(
                        resolved.target_call,
                        "toolbox_load_invalid",
                        "Only the first valid toolbox load in one response may succeed.",
                    )
                else:
                    result = self._toolbox_load(
                        resolved.target_call,
                        projection,
                        sensitivity=sensitivity,
                    )
                resolved = replace(resolved, control_result=result)
                if not result.is_error:
                    load_succeeded = True
            resolved_calls.append(resolved)
        return tuple(resolved_calls)

    def _resolve_call(
        self,
        call: ToolCall,
        projection: StepToolProjection,
    ) -> _ResolvedCall:
        callable_entries = {
            entry.view.name: entry for entry in projection.callable_entries
        }
        entry = callable_entries.get(call.name)
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
        if call.name == "toolbox_load":
            requested_names = call.arguments.get("tool_names")
            if (
                isinstance(requested_names, (tuple, list))
                and len(requested_names) > self._limits.max_loaded_tools
            ):
                return _ResolvedCall(
                    call,
                    call,
                    failure=_error(
                        call,
                        "toolbox_load_limit_exceeded",
                        "The requested on-demand working set exceeds its count bound.",
                    ),
                )
        try:
            arguments = validate_tool_schema_value(
                definition.input_schema,
                call.arguments,
            )
        except (TypeError, ValueError, RuntimeError):
            code = f"{call.name}_invalid"
            return _ResolvedCall(
                call,
                call,
                failure=_error(
                    call,
                    code,
                    "The toolbox control arguments are invalid.",
                ),
            )
        return _ResolvedCall(
            call,
            replace(call, arguments=arguments),
            control_name=call.name,
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
            started = (
                asyncio.get_running_loop().time()
                if self._observer is not None
                else None
            )
            self._emit_tool_started(
                run,
                call,
                capability,
                catalog_entry=resolved.entry,
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
                catalog_entry=resolved.entry,
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
        self._emit_tool_started(run, call, None, catalog_entry=None)
        if prepared_result is not None:
            result = prepared_result
        elif control_name == "toolbox_search":
            result = self._toolbox_search(call, projection, sensitivity=sensitivity)
        elif control_name == "toolbox_inspect":
            result = await self._toolbox_inspect(
                run, call, projection, sensitivity=sensitivity
            )
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
            catalog_entry=None,
        )
        return result

    async def _toolbox_inspect(
        self,
        run: RunInput,
        call: ToolCall,
        projection: StepToolProjection,
        *,
        sensitivity: ModelSensitivity,
    ) -> ToolResultBlock:
        name = call.arguments["tool_name"]
        entry = next(
            (item for item in projection.catalog_entries if item.view.name == name),
            None,
        )
        if entry is None:
            return _error(
                call,
                "toolbox_tool_not_available",
                "The tool is outside this run's prepared candidates.",
            )
        if not _entry_resolves_exactly(entry, self._registry):
            return _error(
                call,
                "toolbox_inspect_stale",
                "The prepared contract no longer resolves exactly.",
            )
        # Local admission only: domain projection performs no executor or remote call.
        # Intersect with the frozen candidates; later attachment cannot expand inspection.
        try:
            current_names = await self._domains[entry.domain_owner_id].project(run)
        except Exception:
            return _error(
                call,
                "toolbox_inspect_unavailable",
                "Current local admission could not be checked; no contract was returned.",
            )
        if name not in current_names:
            return _error(
                call,
                "toolbox_inspect_stale",
                "The prepared tool is no longer locally applicable.",
            )
        contract = _tool_contract(entry)
        expected = call.arguments.get("contract_digest")
        path = cast(str, call.arguments.get("path", ""))
        offset = cast(int, call.arguments.get("offset", 0))
        if expected is not None and expected != contract["contract_digest"]:
            return _error(
                call,
                "toolbox_inspect_stale",
                "The requested contract digest does not match this prepared tool.",
            )
        if (path or offset) and expected is None:
            return _error(
                call,
                "toolbox_inspect_reference_required",
                "Subtree and page retrieval require the exact returned contract_digest.",
            )
        try:
            data = _inspection_page(
                contract,
                path,
                offset,
                maximum_bytes=self._limits.max_toolbox_load_result_bytes,
                maximum_children=self._limits.max_toolbox_search_results,
                maximum_depth=self._limits.max_tool_result_depth - 1,
            )
        except KeyError:
            return _error(
                call,
                "toolbox_inspect_path_invalid",
                "The exact contract path or offset does not exist.",
            )
        except ValueError:
            return _error(
                call,
                "toolbox_inspect_limit_exceeded",
                "The requested inspection cannot fit the configured result bound.",
            )
        return _control_success(
            call,
            "toolbox_inspection_result",
            data,
            sensitivity=max(
                (sensitivity, entry.view.presentation_sensitivity),
                key=lambda item: item.routing_rank,
            ),
            run_id=projection.run_id,
            catalog_digest=projection.catalog_digest,
        )

    def _toolbox_search(
        self,
        call: ToolCall,
        projection: StepToolProjection,
        *,
        sensitivity: ModelSensitivity,
    ) -> ToolResultBlock:
        query = call.arguments.get("query")
        limit_value = call.arguments.get(
            "limit", min(5, self._limits.max_toolbox_search_results)
        )
        if (
            not isinstance(query, str)
            or not isinstance(limit_value, int)
            or isinstance(limit_value, bool)
        ):
            return _error(
                call,
                "toolbox_search_invalid",
                "The toolbox search arguments are invalid.",
            )
        loaded_names = {entry.view.name for entry in projection.loaded_entries}
        scored: list[tuple[int, str, str, RunToolCatalogEntry]] = []
        for entry in projection.catalog_entries:
            score = _toolbox_search_score(query, entry)
            scored.append((score, entry.toolbox_id.value, entry.view.name, entry))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        total = len(scored)
        cursor = call.arguments.get("cursor")
        position = 0
        if cursor is not None:
            try:
                if (
                    not isinstance(cursor, str)
                    or re.fullmatch(r"[1-9][0-9]{0,5}\.[0-9a-f]{64}", cursor) is None
                ):
                    raise ValueError
                position = int(cursor.partition(".")[0])
                expected = self._toolbox_search_cursor(projection, query, position)
                if not hmac.compare_digest(cursor, expected) or position >= total:
                    raise ValueError
            except ValueError:
                return _error(
                    call,
                    "toolbox_search_cursor_invalid",
                    "The search cursor is invalid or belongs to another prepared scope or query.",
                )
        selected = scored[position : position + limit_value]
        matches = [
            _toolbox_search_match(score, entry, loaded_names=loaded_names)
            for score, _, _, entry in selected
        ]
        while True:
            data = {
                "catalog_digest": projection.catalog_digest,
                "matches": matches,
                "total_matches": sum(score > 0 for score, _, _, _ in scored),
                "total_candidates": total,
                "returned_count": len(matches),
                "truncated": position + len(matches) < total,
                "next_cursor": (
                    self._toolbox_search_cursor(
                        projection, query, position + len(matches)
                    )
                    if matches and position + len(matches) < total
                    else None
                ),
            }
            if (
                len(canonical_json(data).encode("utf-8"))
                <= self._limits.max_toolbox_search_result_bytes
                and _json_depth(data) < self._limits.max_tool_result_depth
            ) and (matches or total == 0):
                return _control_success(
                    call,
                    "toolbox_search_result",
                    data,
                    sensitivity=sensitivity,
                    run_id=projection.run_id,
                    catalog_digest=projection.catalog_digest,
                )
            if not matches:
                return _error(
                    call,
                    "toolbox_search_limited",
                    "The bounded toolbox search result cannot fit its byte limit.",
                )
            # Authoring metadata is useful without activating execution schemas,
            # but must not make a candidate unreachable on a bounded page. Omit
            # whole contracts before removing candidates; exact load remains the
            # full-contract inspection path for unusually large declarations.
            expanded = [item for item in matches if "automation_contract" in item]
            if expanded:
                largest = max(
                    expanded,
                    key=lambda item: len(
                        canonical_json(item["automation_contract"]).encode()
                    ),
                )
                del largest["automation_contract"]
                largest["automation_contract_omitted"] = True
                continue
            matches.pop()

    def _toolbox_search_cursor(
        self, projection: StepToolProjection, query: str, position: int
    ) -> str:
        material = canonical_json(
            {
                "run_id": projection.run_id,
                "catalog_digest": projection.catalog_digest,
                "query": query,
                "position": position,
            }
        ).encode("utf-8")
        return (
            f"{position}."
            + hmac.new(self._search_cursor_key, material, "sha256").hexdigest()
        )

    def _toolbox_load(
        self,
        call: ToolCall,
        projection: StepToolProjection,
        *,
        sensitivity: ModelSensitivity,
    ) -> ToolResultBlock:
        names_value = call.arguments.get("tool_names")
        if not isinstance(names_value, (tuple, list)) or any(
            not isinstance(name, str) for name in names_value
        ):
            return _error(
                call,
                "toolbox_load_invalid",
                "The toolbox load requires a bounded distinct list of exact names.",
            )
        names = tuple(names_value)
        entries_by_name = {
            entry.view.name: entry for entry in projection.catalog_entries
        }
        requested: list[RunToolCatalogEntry] = []
        for name in names:
            entry = entries_by_name.get(name)
            if entry is None:
                return _error(
                    call,
                    "toolbox_tool_not_available",
                    "A requested tool is not applicable in this run.",
                    {"tool_name": name},
                )
            if entry.load_mode is not ToolLoadMode.ON_DEMAND:
                return _error(
                    call,
                    "toolbox_load_invalid",
                    "Pinned tools cannot be loaded into the on-demand working set.",
                    {"tool_name": name},
                )
            if not _entry_resolves_exactly(entry, self._registry):
                return _error(
                    call,
                    "toolbox_load_stale",
                    "A requested tool no longer resolves to its frozen run contract.",
                    {"tool_name": name},
                )
            requested.append(entry)
        loaded = tuple(sorted(requested, key=lambda item: item.view.name))
        definitions = tuple(
            self._registry.tool_definition(entry.view.name) for entry in loaded
        )
        definition_bytes = _definition_bytes(definitions)
        if (
            len(loaded) > self._limits.max_loaded_tools
            or definition_bytes > self._limits.max_loaded_tool_definition_bytes
        ):
            return _error(
                call,
                "toolbox_load_limit_exceeded",
                "The requested on-demand working set exceeds its count or byte bound.",
            )
        pinned = tuple(
            entry
            for entry in projection.catalog_entries
            if entry.load_mode is ToolLoadMode.PINNED
        )
        controls = _control_definitions(self._limits, projection.catalog_entries)
        next_definitions = tuple(
            sorted(
                (
                    *(
                        self._registry.tool_definition(item.view.name)
                        for item in pinned
                    ),
                    *definitions,
                    *controls,
                ),
                key=lambda item: item.name,
            )
        )
        try:
            _validate_step_surface(next_definitions, self._limits)
        except ToolSurfaceLimitExceeded:
            return _error(
                call,
                "toolbox_load_limit_exceeded",
                "The complete next-step provider surface exceeds its bound.",
            )
        activation_digest = _activation_digest(
            projection.run_id,
            projection.catalog_digest,
            loaded,
        )
        data = {
            "run_id": projection.run_id,
            "catalog_digest": projection.catalog_digest,
            "loaded_names": [entry.view.name for entry in loaded],
            "definition_bytes": definition_bytes,
            "activation_digest": activation_digest,
            "contracts": [
                _tool_contract(entry, include_schemas=False) for entry in loaded
            ],
        }
        _fit_load_contracts(data, loaded, self._limits)
        if (
            len(canonical_json(data).encode("utf-8"))
            > self._limits.max_toolbox_load_result_bytes
        ):
            return _error(
                call,
                "toolbox_load_limit_exceeded",
                "The exact toolbox load receipt exceeds its byte bound.",
            )
        return _control_success(
            call,
            "toolbox_load_receipt",
            data,
            sensitivity=sensitivity,
            run_id=projection.run_id,
            catalog_digest=projection.catalog_digest,
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
            catalog_entry=entry,
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
            _validate_run_execution_scope(run, capability, sensitivity)
            await self._validate_execution_contracts(run)
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
                source_scope=run.resolved_source_scope,
                request_sensitivity=sensitivity,
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
                    catalog_entry=entry,
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
        result = _with_execution_lineage(result, capability)
        result = _bounded_tool_result(call, result, self._limits)
        self._emit_tool_completed(
            run,
            call,
            result,
            started,
            capability=capability,
            catalog_entry=entry,
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
            source_scope=run.resolved_source_scope,
            request_sensitivity=sensitivity,
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
        _validate_output_execution_scope(run, capability, output)
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
        catalog_entry: RunToolCatalogEntry,
    ) -> tuple[
        ToolResultBlock,
        ToolBatchInterruption | None,
        ToolBatchCertainty,
    ]:
        if capability.operational_effect is OperationalEffect.NONE:
            raise ValueError("effect execution requires an operational effect")
        if capability.effect_receipt_policy is not None:
            if self._effect_receipts is None or self._effect_receipts_unavailable:
                raise CapabilityInputError(
                    "effect_receipt_unavailable",
                    "Durable external-effect evidence is unavailable; no action was dispatched.",
                )
            scope = run.start.execution_scope if run.start is not None else None
            await self._effect_receipts.require_effects_unblocked(
                run.agent_id,
                run_id=run.id,
                routine_id=None if scope is None else scope.routine_id,
            )
            existing = await self._effect_receipts.load_effect_receipt_for_call(
                run.agent_id, run.id, call.id
            )
            if existing is not None:
                return (
                    _effect_duplicate_result(call, existing),
                    None,
                    ToolBatchCertainty.DEFINITE,
                )
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
        if capability.effect_receipt_policy is not None:
            self._validate_effect_plan(run, capability, domain, plan)
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
                catalog_entry=catalog_entry,
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
                    catalog_entry=catalog_entry,
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
                    catalog_entry=catalog_entry,
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
                    catalog_entry=catalog_entry,
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
                catalog_entry=catalog_entry,
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
            await self._validate_execution_contracts(run)
            if capability.effect_receipt_policy is not None:
                current_arguments = await domain.prepare_call(
                    run, call, capability, arguments, request_sensitivity=sensitivity
                )
                if current_arguments != arguments:
                    raise CapabilityInputError(
                        "state_changed",
                        "Current effect binding changed after approval.",
                    )
            if plan.recheck_after_approval:
                try:
                    current = await side_effect.preflight(execution)
                    if not isinstance(current, FrozenJsonObject):
                        raise ValueError(
                            "side-effect preflight must return FrozenJsonObject"
                        )
                    current_plan = await domain.side_effect_plan(
                        run,
                        call,
                        capability,
                        execution,
                        current,
                    )
                    if (
                        capability.effect_receipt_policy is not None
                        and current_plan != plan
                    ):
                        raise CapabilityInputError(
                            "state_changed",
                            "Current effect authorization changed after approval.",
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
            if capability.effect_receipt_policy is not None:
                return await self._execute_reserved_effect(
                    run,
                    call,
                    capability,
                    side_effect,
                    execution,
                    arguments,
                    plan,
                    domain,
                    sensitivity=sensitivity,
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
            _validate_output_execution_scope(run, capability, output)
            if output.artifact is not None or capability.artifact_policy is not None:
                raise ToolOutputValidationError(
                    "side-effect capability cannot produce an artifact draft"
                )
            return (
                _classified_success(call, output),
                interruption_kind,
                outcome_certainty,
            )

    def _validate_effect_plan(
        self,
        run: RunInput,
        capability: Capability,
        domain: CapabilityDomain,
        plan: SideEffectPlan,
    ) -> CapabilityGrant | None:
        if plan.effect_intent is None:
            raise ValueError(
                "receipt-bearing effects require a canonical domain intent"
            )
        scope = run.start.execution_scope if run.start is not None else None
        if scope is None:
            if plan.capability_grant_digest is not None or not plan.approval_required:
                raise ValueError(
                    "foreground external effects require exact per-call approval"
                )
            return None
        grants = tuple(
            grant
            for grant in scope.capability_grants
            if grant.capability_id == capability.id
        )
        if (
            len(grants) != 1
            or plan.approval_required
            or grants[0].grant_digest != plan.capability_grant_digest
        ):
            raise CapabilityInputError(
                "effect_grant_required",
                "The effect has no exact frozen standing grant.",
            )
        grant = grants[0]
        if (
            grant.domain_owner_id != domain.domain_owner_id
            or grant.capability_contract_digest
            != self._registry.contract_digest(capability.id)
            or capability.automation_grant_policy is None
            or grant.constraints_kind
            != capability.automation_grant_policy.constraints_kind
        ):
            raise CapabilityInputError(
                "effect_grant_changed",
                "The retained effect grant contract is no longer current.",
            )
        self._registry.validate_grant_constraints(capability.id, grant.constraints)
        return grant

    async def _execute_reserved_effect(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        executor: SideEffectExecutor,
        execution: ToolExecution,
        arguments: FrozenJsonObject,
        plan: SideEffectPlan,
        domain: CapabilityDomain,
        *,
        sensitivity: ModelSensitivity,
    ) -> tuple[ToolResultBlock, ToolBatchInterruption | None, ToolBatchCertainty]:
        from .storage.sqlite_records import EffectReceipt, effect_receipt_id

        store = self._effect_receipts
        policy = capability.effect_receipt_policy
        assert store is not None and policy is not None
        grant = self._validate_effect_plan(run, capability, domain, plan)
        scope = run.start.execution_scope if run.start is not None else None
        routine_id = None if scope is None else scope.routine_id
        await store.require_effects_unblocked(
            run.agent_id, run_id=run.id, routine_id=routine_id
        )
        operation_key = _sha256_digest(
            {
                "agent_id": run.agent_id,
                "run_id": run.id if routine_id is None else None,
                "routine_id": routine_id,
                "routine_revision": None if scope is None else scope.routine_revision,
                "occurrence_id": None if scope is None else scope.occurrence_id,
                "capability_contract_digest": self._registry.contract_digest(
                    capability.id
                ),
                "effect_intent": plan.effect_intent,
            }
        )
        existing = await store.load_effect_receipt_for_operation(
            run.agent_id, operation_key
        )
        if existing is not None:
            return (
                _effect_duplicate_result(call, existing),
                None,
                ToolBatchCertainty.DEFINITE,
            )
        receipt = EffectReceipt(
            receipt_id=effect_receipt_id(
                agent_id=run.agent_id,
                run_id=run.id,
                call_id=call.id,
                operation_key=operation_key,
            ),
            receipt_kind=policy.receipt_kind,
            agent_id=run.agent_id,
            run_id=run.id,
            call_id=call.id,
            capability_id=capability.id,
            domain_owner_id=domain.domain_owner_id,
            capability_contract_digest=self._registry.contract_digest(capability.id),
            operation_key=operation_key,
            argument_fingerprint=_sha256_digest(arguments),
            sensitivity=sensitivity,
            started_at=self._clock(),
            routine_id=routine_id,
            routine_revision=None if scope is None else scope.routine_revision,
            occurrence_id=None if scope is None else scope.occurrence_id,
            capability_grant_digest=None if grant is None else grant.grant_digest,
        )
        active_task = asyncio.current_task()
        cancellations_before_reservation = (
            0 if active_task is None else active_task.cancelling()
        )
        await store.start_effect_receipt(
            receipt,
            grant=grant,
            max_receipts_per_run=min(self._limits.max_tool_calls_per_run, 256),
        )
        execution = replace(execution, effect_receipt_id=receipt.receipt_id)
        observation: EffectObservation | None = None
        interruption: ToolBatchInterruption | None = None
        certainty = ToolBatchCertainty.DEFINITE
        if (
            active_task is not None
            and active_task.cancelling() > cancellations_before_reservation
        ):
            interruption = ToolBatchInterruption.CANCELLED
            observation = EffectObservation(
                EffectOutcome.NOT_APPLIED, EffectEvidenceBasis.LOCAL_NOT_DISPATCHED
            )
            result = _error(
                call,
                "effect_not_dispatched",
                "The effect was cancelled after reservation and before dispatch.",
            )
        else:
            try:
                candidate, execution_error, interruption, certainty = (
                    await _execute_definitely(
                        executor,
                        execution,
                        recovery_timeout_seconds=self._side_effect_recovery_timeout_seconds,
                    )
                )
                if execution_error is not None:
                    failure = domain.normalize_error(call, execution_error)
                    observation = (
                        None if failure is None else failure.effect_observation
                    )
                    result = self._exception_result(call, execution_error, domain)
                else:
                    if not isinstance(candidate, ToolOutput):
                        raise ToolOutputValidationError(
                            "executor did not return ToolOutput"
                        )
                    observation = candidate.effect_observation
                    if observation is not None:
                        observation = self._registry.validate_effect_observation(
                            capability.id, observation
                        )
                    output = await domain.finalize_output(
                        run,
                        call,
                        capability,
                        arguments,
                        candidate,
                        request_sensitivity=sensitivity,
                    )
                    output = self._registry.validate_output(capability.id, output)
                    _validate_output_execution_scope(run, capability, output)
                    if (
                        output.effect_observation != observation
                        or output.artifact is not None
                    ):
                        raise ToolOutputValidationError(
                            "effect finalization changed its observation or produced an artifact"
                        )
                    result = _classified_success(call, output)
                    result = _bounded_tool_result(
                        call, _with_execution_lineage(result, capability), self._limits
                    )
            except BaseException as error:
                if isinstance(error, asyncio.CancelledError):
                    interruption = _cancel_interruption(error)
                result = self._exception_result(call, error, domain)
        try:
            if observation is not None:
                observation = self._registry.validate_effect_observation(
                    capability.id, observation
                )
        except (ValueError, TypeError, ToolOutputValidationError):
            observation = None
        if observation is None or (
            result.is_error
            and observation.outcome is EffectOutcome.SUCCEEDED
            and observation.evidence_basis is not EffectEvidenceBasis.ADAPTER_VERIFIED
        ):
            observation = EffectObservation(
                EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN
            )
        if observation.outcome is EffectOutcome.UNCERTAIN:
            certainty = ToolBatchCertainty.OUTCOME_UNKNOWN
            result = _error(
                call,
                "effect_outcome_uncertain",
                "The external operation may have acted. It will not be dispatched again; explicit foreground recovery is required.",
            )
        elif observation.outcome is not EffectOutcome.SUCCEEDED and not result.is_error:
            result = _error(
                call, "effect_not_applied", "The external operation was not applied."
            )
        terminal = receipt.finish(
            observation, finished_at=max(self._clock(), receipt.started_at)
        )
        if (
            _tool_result_bound_issue(
                _with_effect_reference(result, terminal), self._limits
            )
            is not None
        ):
            result = _error(
                call,
                "tool_result_too_large",
                "The effect result exceeded its fixed bound.",
            )
            if (
                observation.outcome is EffectOutcome.SUCCEEDED
                and observation.evidence_basis
                is not EffectEvidenceBasis.ADAPTER_VERIFIED
            ):
                certainty = ToolBatchCertainty.OUTCOME_UNKNOWN
                observation = EffectObservation(
                    EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN
                )
                terminal = receipt.finish(
                    observation, finished_at=terminal.finished_at or receipt.started_at
                )
        try:
            terminal = await store.finish_effect_receipt(terminal)
        except BaseException as error:
            self._effect_receipts_unavailable = True
            if isinstance(error, asyncio.CancelledError):
                interruption = _cancel_interruption(error)
            result = _error(
                call,
                "effect_receipt_unavailable",
                "The terminal receipt could not be persisted. External effects are blocked; no automatic retry is allowed.",
            )
            return (
                _with_effect_reference(result, receipt),
                interruption,
                ToolBatchCertainty.OUTCOME_UNKNOWN,
            )
        return _with_effect_reference(result, terminal), interruption, certainty

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
        if draft.provenance.authorship not in policy.allowed_authorships:
            raise ToolOutputValidationError(
                "artifact draft authorship is outside the capability policy"
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
        from .storage.sqlite_records import (
            EffectReceiptConflictError,
            EffectUnresolvedError,
        )

        if isinstance(error, EffectUnresolvedError):
            return _error(
                call,
                "effect_unresolved",
                str(error),
                {
                    "receipt_ids": error.receipt_ids,
                    "omitted_count": error.omitted_count,
                },
            )
        if isinstance(error, EffectReceiptConflictError):
            return _error(call, "effect_reservation_conflict", str(error))
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
        catalog_entry: RunToolCatalogEntry | None,
    ) -> None:
        data: dict[str, object] = {"call_id": call.id, "tool_name": call.name}
        if capability is not None:
            data["capability_id"] = capability.id
        if catalog_entry is not None:
            data.update(_toolbox_observation(catalog_entry))
        self._emit(AgentEventKind.TOOL_STARTED, run, data)

    def _emit_approval_requested(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        *,
        catalog_entry: RunToolCatalogEntry,
    ) -> None:
        self._emit(
            AgentEventKind.APPROVAL_REQUESTED,
            run,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "capability_id": capability.id,
                **_toolbox_observation(catalog_entry),
            },
        )

    def _emit_approval_decided(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        outcome: str,
        *,
        catalog_entry: RunToolCatalogEntry,
    ) -> None:
        self._emit(
            AgentEventKind.APPROVAL_DECIDED,
            run,
            {
                "call_id": call.id,
                "tool_name": call.name,
                "capability_id": capability.id,
                **_toolbox_observation(catalog_entry),
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
        catalog_entry: RunToolCatalogEntry | None,
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
        if catalog_entry is not None:
            data.update(_toolbox_observation(catalog_entry))
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
                run_origin=run.origin.value,
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


def _validate_pinned_surface(
    definitions: tuple[ToolDefinition, ...],
    limits: LoopLimits,
) -> None:
    definition_bytes = _definition_bytes(definitions)
    if (
        len(definitions) > limits.max_pinned_tools
        or definition_bytes > limits.max_pinned_tool_definition_bytes
    ):
        raise ToolSurfaceLimitExceeded(
            observed_tools=len(definitions),
            maximum_tools=limits.max_pinned_tools,
            observed_definition_bytes=definition_bytes,
            maximum_definition_bytes=limits.max_pinned_tool_definition_bytes,
        )


def _validate_step_surface(
    definitions: tuple[ToolDefinition, ...],
    limits: LoopLimits,
) -> None:
    definition_bytes = _definition_bytes(definitions)
    if (
        len(definitions) > limits.max_step_tools
        or definition_bytes > limits.max_step_tool_definition_bytes
    ):
        raise ToolSurfaceLimitExceeded(
            observed_tools=len(definitions),
            maximum_tools=limits.max_step_tools,
            observed_definition_bytes=definition_bytes,
            maximum_definition_bytes=limits.max_step_tool_definition_bytes,
        )


def _entry_resolves_exactly(
    entry: RunToolCatalogEntry,
    registry: CapabilityRegistry,
) -> bool:
    try:
        view, capability, owner_id = registry.resolve_tool_owner(entry.view.name)
    except KeyError:
        return False
    return all(
        (
            view == entry.view,
            capability == entry.capability,
            owner_id == entry.domain_owner_id,
            capability.executor_id == entry.executor_id,
            _sha256_digest(capability.input_schema) == entry.input_schema_digest,
            view.presentation.toolbox_id is entry.toolbox_id,
            view.presentation.load_mode is entry.load_mode,
        )
    )


def _catalog_entry_material(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "domain_owner_id": entry.domain_owner_id,
        "executor_id": entry.executor_id,
        "description": entry.view.description,
        "presentation_sensitivity": entry.view.presentation_sensitivity.value,
        "connector_presentation": entry.view.connector_presentation,
        "input_schema": entry.capability.input_schema,
        "input_schema_digest": entry.input_schema_digest,
        "output_kind": entry.capability.output_kind,
        "data_access": entry.capability.access_mode.value,
        "operational_effect": entry.capability.operational_effect.value,
        "presentation": {
            "toolbox_id": entry.toolbox_id.value,
            "load_mode": entry.load_mode.value,
            "text_trust": entry.view.presentation.text_trust.value,
            "summary": entry.view.presentation.summary,
            "when_to_use": entry.view.presentation.when_to_use,
            "keywords": entry.view.presentation.keywords,
        },
        "origin_revision_digest": entry.origin_revision_digest,
        "automation_contract": _automation_contract(entry),
    }


def _automation_contract(entry: RunToolCatalogEntry) -> dict[str, object]:
    """Bounded declaration data for authoring, never an execution grant."""
    policy = entry.capability.automation_grant_policy
    receipt = entry.capability.effect_receipt_policy
    return {
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "automation_eligibility": entry.capability.automation_eligibility.value,
        "requires_automation_grant": policy is not None,
        "grant_policy": (
            None
            if policy is None
            else {
                "constraints_kind": policy.constraints_kind,
                "constraints_schema": policy.constraints_schema,
            }
        ),
        "effect_evidence_basis": (
            None if receipt is None else receipt.success_evidence_basis.value
        ),
        "connector": entry.view.connector_presentation,
    }


def _tool_contract_reference(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "contract_digest": _sha256_digest(
            {
                "entry": _catalog_entry_material(entry),
                "output_schema": entry.capability.output_schema,
            }
        ),
        "input_schema_digest": entry.input_schema_digest,
        "origin_revision_digest": entry.origin_revision_digest,
        "complete": False,
        "inspection_tool": "toolbox_inspect",
    }


def _tool_contract(
    entry: RunToolCatalogEntry, *, include_schemas: bool = True
) -> dict[str, object]:
    """One registered declaration projection shared by search, load and inspection."""
    contract = {
        **_automation_contract(entry),
        **_tool_contract_reference(entry),
    }
    if include_schemas:
        contract.update(
            complete=True,
            input_schema=entry.capability.input_schema,
            output_schema=entry.capability.output_schema,
            output_kind=entry.capability.output_kind,
            data_access=entry.capability.access_mode.value,
            operational_effect=entry.capability.operational_effect.value,
        )
    return contract


def _fit_load_contracts(
    data: dict[str, object],
    entries: tuple[RunToolCatalogEntry, ...],
    limits: LoopLimits,
) -> None:
    """Omit whole schemas deterministically; verification uses the same projection."""
    contracts = cast(list[dict[str, object]], data["contracts"])
    remaining = list(range(len(contracts)))
    while remaining and (
        len(canonical_json(data).encode("utf-8")) > limits.max_toolbox_load_result_bytes
        or _json_depth(data) >= limits.max_tool_result_depth
    ):
        index = max(
            remaining,
            key=lambda i: (
                (
                    _json_depth(contracts[i])
                    if _json_depth(data) >= limits.max_tool_result_depth
                    else 0
                ),
                len(canonical_json(contracts[i]).encode("utf-8")),
            ),
        )
        contracts[index] = _tool_contract_reference(entries[index])
        remaining.remove(index)


def _inspection_page(
    contract: dict[str, object],
    path: str,
    offset: int,
    *,
    maximum_bytes: int,
    maximum_children: int,
    maximum_depth: int,
) -> dict[str, object]:
    """Read a bounded part of one immutable contract; paths cannot address state."""
    value: object = contract
    if path:
        if not path.startswith("/"):
            raise KeyError(path)
        for encoded in path[1:].split("/"):
            if re.search(r"~(?![01])", encoded):
                raise KeyError(path)
            token = encoded.replace("~1", "/").replace("~0", "~")
            if isinstance(value, Mapping) and token in value:
                value = value[token]
            elif isinstance(value, (tuple, list)) and re.fullmatch(
                r"0|[1-9][0-9]*", token
            ):
                index = int(token)
                if index >= len(value):
                    raise KeyError(path)
                value = value[index]
            else:
                raise KeyError(path)
    base = {
        "tool_name": contract["tool_name"],
        "contract_digest": contract["contract_digest"],
        "path": path,
        "offset": offset,
    }
    full = {**base, "complete": True, "value": value}
    if (
        offset == 0
        and len(canonical_json(full).encode("utf-8")) <= maximum_bytes
        and _json_depth(full) <= maximum_depth
    ):
        return full
    if isinstance(value, str):
        if offset >= len(value):
            raise KeyError(path)
        end = min(len(value), offset + maximum_bytes)
        while end > offset:
            page = {
                **base,
                "complete": False,
                "value_type": "string",
                "text": value[offset:end],
                "next_offset": end if end < len(value) else None,
                "total_characters": len(value),
            }
            if (
                len(canonical_json(page).encode("utf-8")) <= maximum_bytes
                and _json_depth(page) <= maximum_depth
            ):
                return page
            end = offset + (end - offset) // 2
        raise ValueError("inspection string fragment cannot fit")
    if isinstance(value, Mapping):
        children = sorted(value.items())
        kind = "object"
    elif isinstance(value, (tuple, list)):
        children = [(str(index), child) for index, child in enumerate(value)]
        kind = "array"
    else:
        raise ValueError("inspection scalar cannot fit")
    if offset >= len(children):
        raise KeyError(path)
    items: list[dict[str, object]] = []
    page = {
        **base,
        "complete": False,
        "value_type": kind,
        "children": items,
        "total_children": len(children),
        "next_offset": offset,
    }
    for key, child in children[offset : offset + maximum_children]:
        child_path = path + "/" + key.replace("~", "~0").replace("/", "~1")
        item = {"path": child_path, "complete": True, "value": child}
        items.append(item)
        page["next_offset"] = (
            offset + len(items) if offset + len(items) < len(children) else None
        )
        if (
            len(canonical_json(page).encode("utf-8")) > maximum_bytes
            or _json_depth(page) > maximum_depth
        ):
            items[-1] = {"path": child_path, "complete": False}
        if (
            len(canonical_json(page).encode("utf-8")) > maximum_bytes
            or _json_depth(page) > maximum_depth
        ):
            items.pop()
            page["next_offset"] = offset + len(items)
            break
    if not items:
        raise ValueError("inspection child reference cannot fit")
    return page


def _toolbox_manifest(
    entries: tuple[RunToolCatalogEntry, ...],
) -> tuple[ToolboxManifestEntry, ...]:
    available = {entry.toolbox_id for entry in entries}
    return tuple(
        ToolboxManifestEntry(
            toolbox_id=definition.id,
            label=definition.label,
            summary=definition.summary,
            pinned_count=sum(
                entry.toolbox_id is definition.id
                and entry.load_mode is ToolLoadMode.PINNED
                for entry in entries
            ),
            on_demand_count=sum(
                entry.toolbox_id is definition.id
                and entry.load_mode is ToolLoadMode.ON_DEMAND
                for entry in entries
            ),
            access_modes=tuple(
                sorted(
                    {
                        entry.capability.access_mode
                        for entry in entries
                        if entry.toolbox_id is definition.id
                    },
                    key=lambda mode: mode.value,
                )
            ),
            operational_effects=tuple(
                sorted(
                    {
                        entry.capability.operational_effect
                        for entry in entries
                        if entry.toolbox_id is definition.id
                    },
                    key=lambda effect: effect.value,
                )
            ),
        )
        for definition in TOOLBOX_DEFINITIONS
        if definition.id in available
    )


def _manifest_material(
    manifest: tuple[ToolboxManifestEntry, ...],
) -> list[dict[str, object]]:
    return [
        {
            "toolbox_id": item.toolbox_id.value,
            "label": item.label,
            "summary": item.summary,
            "pinned_count": item.pinned_count,
            "on_demand_count": item.on_demand_count,
            "access_modes": tuple(mode.value for mode in item.access_modes),
            "operational_effects": tuple(
                effect.value for effect in item.operational_effects
            ),
        }
        for item in manifest
    ]


def _loaded_entry_material(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "tool_name": entry.view.name,
        "capability_id": entry.capability.id,
        "domain_owner_id": entry.domain_owner_id,
        "executor_id": entry.executor_id,
        "input_schema_digest": entry.input_schema_digest,
        "origin_revision_digest": entry.origin_revision_digest,
        "toolbox_id": entry.toolbox_id.value,
        "load_mode": entry.load_mode.value,
        "text_trust": entry.view.presentation.text_trust.value,
    }


def _activation_digest(
    run_id: str,
    catalog_digest: str,
    entries: tuple[RunToolCatalogEntry, ...],
) -> str:
    return _sha256_digest(
        {
            "run_id": run_id,
            "catalog_digest": catalog_digest,
            "loaded_tools": [_loaded_entry_material(entry) for entry in entries],
        }
    )


def _loaded_tool_receipt(
    *,
    run_id: str,
    catalog_digest: str,
    catalog_entries: tuple[RunToolCatalogEntry, ...],
    messages: tuple[object, ...],
    limits: LoopLimits,
    registry: CapabilityRegistry,
) -> tuple[tuple[RunToolCatalogEntry, ...], str, str]:
    exchanges = _ordered_tool_exchanges(messages)
    loaded: tuple[RunToolCatalogEntry, ...] = ()
    activation_digest = _activation_digest(run_id, catalog_digest, ())
    for call, block in exchanges:
        if call.name != "toolbox_load" or block.is_error:
            continue
        names = call.arguments.get("tool_names")
        if not isinstance(names, (tuple, list)) or not all(
            isinstance(name, str) for name in names
        ):
            continue
        if block.output.get("kind") != "toolbox_load_receipt":
            continue
        data = block.output.get("data")
        provenance = block.sensitivity_provenance
        if not isinstance(data, Mapping) or (
            provenance.get("authority") != "tool_catalog_control"
            or provenance.get("run_id") != run_id
            or provenance.get("catalog_digest") != catalog_digest
            or provenance.get("control_name") != "toolbox_load"
        ):
            continue
        candidate = _verified_loaded_entries(
            data,
            requested_names=tuple(names),
            run_id=run_id,
            catalog_digest=catalog_digest,
            catalog_entries=catalog_entries,
            limits=limits,
            registry=registry,
        )
        if candidate is None:
            continue
        loaded = candidate
        activation_digest = _activation_digest(run_id, catalog_digest, loaded)
    return loaded, activation_digest, _toolbox_exchanges_digest(exchanges)


def _ordered_tool_exchanges(
    messages: tuple[object, ...],
) -> tuple[tuple[ToolCall, ToolResultBlock], ...]:
    pending: list[ToolCall] = []
    exchanges: list[tuple[ToolCall, ToolResultBlock]] = []
    for message in messages:
        if not isinstance(message, CanonicalMessage):
            raise TypeError("step transcript must contain CanonicalMessage records")
        if message.role is MessageRole.ASSISTANT:
            if pending:
                raise ValueError("step transcript has incomplete ordered tool results")
            pending = list(message.tool_calls)
            continue
        if message.role is MessageRole.TOOL:
            for block in message.content:
                if not isinstance(block, ToolResultBlock):
                    raise TypeError("tool message must contain ToolResultBlock records")
                if not pending:
                    raise ValueError(
                        "step transcript contains an unexpected tool result"
                    )
                call = pending.pop(0)
                if block.call_id != call.id:
                    raise ValueError("step transcript tool results are out of order")
                exchanges.append((call, block))
            continue
        if pending:
            raise ValueError("step transcript has incomplete ordered tool results")
    if pending:
        raise ValueError("step transcript has incomplete ordered tool results")
    return tuple(exchanges)


def _toolbox_transcript_digest(messages: tuple[object, ...]) -> str:
    """Digest ordered toolbox-load exchanges without treating call IDs as global."""

    return _toolbox_exchanges_digest(_ordered_tool_exchanges(messages))


def _toolbox_exchanges_digest(
    exchanges: tuple[tuple[ToolCall, ToolResultBlock], ...],
) -> str:
    return _sha256_digest(
        [
            {
                "call": {
                    "id": call.id,
                    "provider_call_id": call.provider_call_id,
                    "name": call.name,
                    "arguments": call.arguments,
                },
                "result": {
                    "call_id": result.call_id,
                    "output": result.output,
                    "is_error": result.is_error,
                    "sensitivity": (
                        None if result.sensitivity is None else result.sensitivity.value
                    ),
                    "sensitivity_provenance": result.sensitivity_provenance,
                    "capability_id": result.capability_id,
                    "executor_id": result.executor_id,
                },
            }
            for call, result in exchanges
            if call.name == "toolbox_load"
        ]
    )


def _verified_loaded_entries(
    data: Mapping[str, object],
    *,
    requested_names: tuple[str, ...],
    run_id: str,
    catalog_digest: str,
    catalog_entries: tuple[RunToolCatalogEntry, ...],
    limits: LoopLimits,
    registry: CapabilityRegistry,
) -> tuple[RunToolCatalogEntry, ...] | None:
    if set(data) != {
        "run_id",
        "catalog_digest",
        "loaded_names",
        "definition_bytes",
        "activation_digest",
        "contracts",
    }:
        return None
    names_value = data.get("loaded_names")
    if not isinstance(names_value, (tuple, list)):
        return None
    names = tuple(name for name in names_value if isinstance(name, str))
    if (
        len(names) != len(names_value)
        or names != tuple(sorted(requested_names))
        or len(names) != len(set(names))
        or len(names) > limits.max_loaded_tools
        or data.get("run_id") != run_id
        or data.get("catalog_digest") != catalog_digest
    ):
        return None
    by_name = {entry.view.name: entry for entry in catalog_entries}
    try:
        entries = tuple(by_name[name] for name in names)
    except KeyError:
        return None
    if any(
        entry.load_mode is not ToolLoadMode.ON_DEMAND
        or not _entry_resolves_exactly(entry, registry)
        for entry in entries
    ):
        return None
    definitions = tuple(registry.tool_definition(entry.view.name) for entry in entries)
    definition_bytes = _definition_bytes(definitions)
    expected_activation = _activation_digest(run_id, catalog_digest, entries)
    expected_data: dict[str, object] = dict(data)
    expected_data["contracts"] = [
        _tool_contract(entry, include_schemas=False) for entry in entries
    ]
    _fit_load_contracts(expected_data, entries, limits)
    if (
        data.get("definition_bytes") != definition_bytes
        or canonical_json(data.get("contracts"))
        != canonical_json(expected_data["contracts"])
        or definition_bytes > limits.max_loaded_tool_definition_bytes
        or data.get("activation_digest") != expected_activation
        or len(canonical_json(data).encode("utf-8"))
        > limits.max_toolbox_load_result_bytes
    ):
        return None
    return entries


def _toolbox_search_score(query: str, entry: RunToolCatalogEntry) -> int:
    normalized = query.strip().lower()
    terms = tuple(_TOKEN.findall(normalized))
    if not terms:
        return 0
    score = 0
    if normalized == entry.view.name:
        score += 10_000
    phrases = (
        entry.view.name.replace("_", " "),
        entry.view.presentation.summary.lower(),
        entry.view.presentation.when_to_use.lower(),
        *entry.view.presentation.keywords,
    )
    if normalized in phrases:
        score += 8_000
    weighted_fields = (
        (entry.view.name, 100),
        (entry.capability.id, 50),
        (entry.view.presentation.summary, 20),
        (entry.view.presentation.when_to_use, 10),
        (" ".join(entry.view.presentation.keywords), 30),
        (" ".join(entry.parameter_names), 15),
        (
            (
                canonical_json(entry.view.connector_presentation)
                if entry.view.connector_presentation is not None
                else ""
            ),
            20,
        ),
    )
    for text, weight in weighted_fields:
        tokens = set(_TOKEN.findall(text.lower()))
        score += weight * sum(term in tokens for term in terms)
    return score


def _toolbox_search_match(
    score: int,
    entry: RunToolCatalogEntry,
    *,
    loaded_names: set[str],
) -> dict[str, object]:
    load_state = (
        ToolLoadMode.PINNED.value
        if entry.load_mode is ToolLoadMode.PINNED
        else ("loaded" if entry.view.name in loaded_names else "on_demand")
    )
    match: dict[str, object] = {
        "tool_name": entry.view.name,
        "toolbox_id": entry.toolbox_id.value,
        "capability_id": entry.capability.id,
        "automation_eligibility": entry.capability.automation_eligibility.value,
        "requires_automation_grant": entry.capability.automation_grant_policy
        is not None,
        "summary": entry.view.presentation.summary,
        "when_to_use": entry.view.presentation.when_to_use,
        "text_trust": entry.view.presentation.text_trust.value,
        "load_state": load_state,
        "data_access": entry.capability.access_mode.value,
        "operational_effect": entry.capability.operational_effect.value,
        "score": score,
        "match_status": "matched" if score > 0 else "unmatched_fallback",
    }
    if entry.capability.automation_grant_policy is not None:
        match["automation_contract"] = _tool_contract(entry)
    match["inspection_tool"] = "toolbox_inspect"
    return match


def _projection_digest(
    *,
    run_id: str,
    registry_digest: str,
    catalog_digest: str,
    transcript_digest: str,
    provider_definitions: tuple[ToolDefinition, ...],
    callable_entries: tuple[RunToolCatalogEntry, ...],
    loaded_entries: tuple[RunToolCatalogEntry, ...],
    activation_digest: str,
    source_scope: EffectiveSourceScope | None = None,
) -> str:
    return _sha256_digest(
        {
            "run_id": run_id,
            "registry_digest": registry_digest,
            "catalog_digest": catalog_digest,
            "transcript_digest": transcript_digest,
            "provider_definitions": [
                _definition_material(item) for item in provider_definitions
            ],
            "callable_tools": [entry.view.name for entry in callable_entries],
            "loaded_tools": [entry.view.name for entry in loaded_entries],
            "activation_digest": activation_digest,
            "source_scope": None if source_scope is None else source_scope.to_mapping(),
        }
    )


def _step_projection_digest(projection: StepToolProjection) -> str:
    """Recompute one projection digest from its exact runtime-owned material."""

    if not isinstance(projection, StepToolProjection):
        raise TypeError("projection must be StepToolProjection")
    return _projection_digest(
        run_id=projection.run_id,
        registry_digest=projection.registry_digest,
        catalog_digest=projection.catalog_digest,
        transcript_digest=projection.transcript_digest,
        provider_definitions=projection.provider_definitions,
        callable_entries=projection.callable_entries,
        loaded_entries=projection.loaded_entries,
        activation_digest=projection.activation_digest,
        source_scope=projection.source_scope,
    )


def _validate_step_projection(
    projection: StepToolProjection,
    registry: CapabilityRegistry,
    limits: LoopLimits,
) -> None:
    if projection.registry_digest != registry.digest:
        raise ValueError("step projection registry identity changed")
    catalog_names = [entry.view.name for entry in projection.catalog_entries]
    if len(catalog_names) != len(set(catalog_names)) or any(
        not _entry_resolves_exactly(entry, registry)
        for entry in projection.catalog_entries
    ):
        raise ValueError("step projection catalog entries do not resolve exactly")
    pinned = tuple(
        entry
        for entry in projection.catalog_entries
        if entry.load_mode is ToolLoadMode.PINNED
    )
    loaded = tuple(projection.loaded_entries)
    if any(
        entry not in projection.catalog_entries
        or entry.load_mode is not ToolLoadMode.ON_DEMAND
        for entry in loaded
    ) or len({entry.view.name for entry in loaded}) != len(loaded):
        raise ValueError("step projection loaded set is invalid")
    expected_callable = tuple(
        sorted((*pinned, *loaded), key=lambda item: item.view.name)
    )
    if projection.callable_entries != expected_callable:
        raise ValueError("step projection callable set is invalid")
    loaded_definitions = tuple(
        registry.tool_definition(entry.view.name) for entry in loaded
    )
    if (
        len(loaded) > limits.max_loaded_tools
        or (
            loaded
            and _definition_bytes(loaded_definitions)
            > limits.max_loaded_tool_definition_bytes
        )
        or projection.loaded_definition_bytes != _definition_bytes(loaded_definitions)
    ):
        raise ValueError("step projection loaded definitions exceed their bounds")
    controls = _control_definitions(limits, projection.catalog_entries)
    expected_definitions = tuple(
        sorted(
            (
                *(
                    registry.tool_definition(entry.view.name)
                    for entry in expected_callable
                ),
                *controls,
            ),
            key=lambda item: item.name,
        )
    )
    if projection.provider_definitions != expected_definitions:
        raise ValueError("step projection provider definitions are invalid")
    _validate_step_surface(expected_definitions, limits)
    expected_activation = _activation_digest(
        projection.run_id, projection.catalog_digest, loaded
    )
    if projection.activation_digest != expected_activation:
        raise ValueError("step projection activation digest is invalid")
    if projection.projection_digest != _step_projection_digest(projection):
        raise ValueError("step projection digest is invalid")


def _toolbox_observation(entry: RunToolCatalogEntry) -> dict[str, object]:
    return {
        "toolbox_id": entry.toolbox_id.value,
        "load_mode": entry.load_mode.value,
        "provider_state": (
            "pinned" if entry.load_mode is ToolLoadMode.PINNED else "loaded"
        ),
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
    if (
        run.origin is RunOrigin.SCHEDULED_ROUTINE
        and capability.automation_eligibility
        is not AutomationEligibility.AUTOMATION_DIRECT
    ):
        raise CapabilityInputError(
            "scheduled_capability_ineligible",
            "The requested capability is not admitted for scheduled execution.",
            {"capability_id": capability.id},
        )
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
    source_id = call.arguments.get("source_id")
    if isinstance(source_id, str):
        return source_id
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


__all__ = [
    "CapabilityDomain",
    "CapabilityFailure",
    "CapabilityRuntime",
    "InternalCapabilityOutcome",
    "InternalCapabilityRequest",
    "RunToolCatalog",
    "RunToolCatalogEntry",
    "SideEffectPlan",
    "StepToolProjection",
    "ToolboxManifestEntry",
]
