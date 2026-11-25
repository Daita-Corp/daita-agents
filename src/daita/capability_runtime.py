"""Validate, govern, execute, and normalize capabilities across static domains."""

from __future__ import annotations

import asyncio
import hmac
import re
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager, nullcontext
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol, cast

from ._json import FrozenJsonObject, canonical_json
from .artifacts.models import (
    ArtifactDraft,
    ArtifactError,
    ArtifactRef,
    canonical_artifact_filename,
)
from .artifacts.store import AgentHomeArtifactStore
from .capabilities import (
    DEFAULT_APPROVAL_REASON,
    RESERVED_TOOL_NAMES,
    TOOLBOX_DEFINITIONS,
    AccessMode,
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    CallProvenance,
    CallTargetLane,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    CapabilityRegistry,
    EffectEvidenceBasis,
    EffectObservation,
    EffectOutcome,
    ExecutionAdmissionPolicy,
    ExecutionContractBindings,
    ExecutionContractReader,
    ExecutionPreference,
    Executor,
    MachineRunDirective,
    MachineRunDirectiveKind,
    OperationalEffect,
    RecoveryKind,
    SideEffectExecutor,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolOutputValidationError,
    ToolRecovery,
    ToolView,
    model_identity_key,
    omit_unset_arguments,
    recovery_details,
    tool_call_mode,
    tool_route_arguments,
    tool_route_unknown_arguments,
    validate_tool_schema_value,
)
from .capability_domain_calls import (
    _domain_call_provenance,
    _domain_finalize_output,
    _domain_prepare_call,
    _domain_project,
    _domain_side_effect_plan,
)
from .capability_programmatic_reads import programmatic_reader
from .capability_results import (
    _BATCH_RESULT_REFS,
    _batch_result_refs,
    _bounded_tool_result,
    _cancel_interruption,
    _classified_success,
    _consume_background_task,
    _consume_read_task,
    _control_success,
    _duration_ms,
    _effect_duplicate_result,
    _error,
    _interrupted_batch,
    _interruption_result,
    _json_depth,
    _result_bytes,
    _result_error_code,
    _settle_cancelled_reads,
    _source_pressure_key,
    _tool_result_bound_issue,
    _validate_output_execution_scope,
    _validate_run_execution_scope,
    _with_effect_reference,
    _with_execution_disposition,
    _with_execution_lineage,
)
from .capability_toolbox import (
    _activation_digest,
    _catalog_entry_material,
    _definition_bytes,
    _entry_definition,
    _entry_resolves_exactly,
    _fit_load_contracts,
    _inspection_page,
    _loaded_tool_receipt,
    _manifest_material,
    _projection_digest,
    _sha256_digest,
    _tool_contract,
    _toolbox_observation,
    _toolbox_search_match,
    _toolbox_search_score,
    _toolbox_transcript_digest,
    _validate_pinned_surface,
    _validate_step_surface,
)
from .errors import DaitaError
from .llm.errors import (
    ToolCatalogLimitExceeded,
    ToolManifestLimitExceeded,
    ToolSurfaceLimitExceeded,
)
from .llm.models import (
    CanonicalMessage,
    ModelSensitivity,
    ToolCall,
    ToolDefinition,
    ToolResultBlock,
    ToolResultRefs,
    tool_result_refs,
)
from .loop.models import (
    LoopLimits,
    RunInput,
    RunOrigin,
    ToolBatchCertainty,
    ToolBatchInterruption,
    ToolBatchOutcome,
)
from .loop.session import RunCancellationToken, RunSession
from .observation import AgentEvent, AgentEventKind, AgentObserver, _emit_safely
from .scope import EffectiveSourceScope

if TYPE_CHECKING:
    from .domains.analysis.domain import AnalysisCapabilityDomain
    from .storage.sqlite_records import EffectReceipt


class AdmissionPermit(Protocol):
    async def __aenter__(self) -> object: ...

    async def __aexit__(self, exc_type, exc, traceback, /) -> bool | None: ...


class CapabilityAdmissionCoordinator(Protocol):
    async def source_resource_permit(
        self,
        key: str,
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> AdmissionPermit: ...

    async def mcp_permit(
        self,
        key: str,
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> AdmissionPermit: ...


class CapabilityEffectCoordinator(Protocol):
    async def acquire(
        self,
        *,
        deadline: float | None = None,
        cancellation: RunCancellationToken | None = None,
    ) -> AdmissionPermit: ...


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
        max_receipts_per_run: int = 64,
    ) -> EffectReceipt: ...
    async def finish_effect_receipt(self, receipt: EffectReceipt) -> EffectReceipt: ...


@dataclass(frozen=True, slots=True)
class MachineRunScope:
    """What a machine run may use, fixed when it starts (invariant 6)."""

    source_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    capability_ids: tuple[str, ...]
    connector_binding_ids: tuple[str, ...]
    contract_bindings: ExecutionContractBindings
    unrestricted_source_ids: tuple[str, ...] = ()


# Builds the scope at run start: (origin, named source ids, model route ids).
MachineScopeBuilder = Callable[
    [RunOrigin, tuple[str, ...], tuple[str, ...]], Awaitable[MachineRunScope]
]


def machine_run_capability_ids(
    registry: CapabilityRegistry, excluded_owners: frozenset[str]
) -> tuple[str, ...]:
    """Every registered capability whose owner is not excluded for this run.

    The composition root owns the exclusion list (invariant 6); the runtime
    stays free of domain identities.
    """

    return tuple(
        sorted(
            capability_id
            for capability_id in registry.capability_ids
            if registry.resolve_domain_owner(capability_id) not in excluded_owners
        )
    )


def machine_run_connector_binding_ids(registry: CapabilityRegistry) -> tuple[str, ...]:
    """Every connector binding behind a registered tool."""

    bindings: set[str] = set()
    for name in registry.tool_names:
        presentation = registry.resolve_tool(name)[0].connector_presentation
        if presentation is not None and isinstance(presentation.get("id"), str):
            bindings.add(cast(str, presentation["id"]))
    return tuple(sorted(bindings))


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
    approval_reason: str = DEFAULT_APPROVAL_REASON
    recheck_after_approval: bool = True
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
        if self.effect_intent is not None:
            if not isinstance(self.effect_intent, FrozenJsonObject):
                raise TypeError(
                    "side-effect intent must be domain-normalized frozen JSON"
                )
            if len(canonical_json(self.effect_intent).encode("utf-8")) > 32 * 1024:
                raise ValueError("side-effect intent exceeds its byte bound")


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
    # For a mode-routed tool: the admitted (mode, entry) routes of this run.
    # The outer entry is the route of the view's primary capability when it
    # is admitted, otherwise the first admitted route.
    routes: tuple[tuple[str, RunToolCatalogEntry], ...] = ()

    def route(self, mode: object) -> RunToolCatalogEntry | None:
        if not self.routes:
            return self
        return next((entry for name, entry in self.routes if name == mode), None)

    @property
    def capabilities(self) -> tuple[Capability, ...]:
        if not self.routes:
            return (self.capability,)
        return tuple(entry.capability for _, entry in self.routes)

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
    call_target_lane: CallTargetLane | None
    source_scope: EffectiveSourceScope | None = None
    session: RunSession | None = field(default=None, repr=False, compare=False)

    @property
    def capability_ids(self) -> frozenset[str]:
        return frozenset(
            capability.id for entry in self.entries for capability in entry.capabilities
        )

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
    call_target_lane: CallTargetLane | None
    source_scope: EffectiveSourceScope | None = None
    session: RunSession | None = field(default=None, repr=False, compare=False)

    @property
    def limits_guidance(self) -> str:
        """One compact line of enforced per-call bounds for callable tools.

        Each bound is the ceiling this run's lane enforces. A durable bound is
        named only for foreground runs with a prepared durable entrypoint that
        extends the foreground bound; execution still rechecks applicability.
        """

        durable = (
            next(
                (
                    entry
                    for entry in self.catalog_entries
                    if entry.capability.durable_foreground_entrypoint
                ),
                None,
            )
            if self.call_target_lane is CallTargetLane.FOREGROUND
            else None
        )
        parts: list[str] = []
        for entry in self.callable_entries:
            policy = entry.capability.execution_admission_policy
            ceiling = (
                None
                if policy is None
                else policy.call_target_ceiling(self.call_target_lane)
            )
            if policy is None or ceiling is None:
                continue
            argument = model_identity_key(cast(str, policy.target_count_argument))
            part = f"{entry.view.name} at most {ceiling} {argument}"
            if (
                durable is not None
                and policy.graph_v1_eligible
                and policy.graph_max_targets > ceiling
            ):
                part += (
                    f" (up to {policy.graph_max_targets} through "
                    f"{durable.view.name}"
                    + (
                        ""
                        if durable in self.callable_entries
                        else " after toolbox_load"
                    )
                    + ")"
                )
            parts.append(part)
        return "" if not parts else "Per-call limits: " + "; ".join(parts) + "."

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


@dataclass(slots=True)
class _DispatchState:
    """Whether one call reached its executor or effect reservation."""

    started: bool = False


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


def call_target_lane(origin: RunOrigin) -> CallTargetLane | None:
    """Return the per-call target lane a run origin is held to.

    Enforcement and every model-facing limit text read the lane through this
    one mapping, so a run is shown exactly the ceiling that admits its calls.
    Foreground user calls use the inline ceiling, job runs and sub-runs the job
    ceiling; other machine origins carry no per-call target ceiling.
    """

    if origin is RunOrigin.USER:
        return CallTargetLane.FOREGROUND
    if origin in {RunOrigin.JOB, RunOrigin.JOB_SUBRUN}:
        return CallTargetLane.JOB
    return None


def _require_call_target_ceiling(
    policy: ExecutionAdmissionPolicy | None,
    lane: CallTargetLane | None,
    arguments: Mapping[str, object],
) -> None:
    """Reject a machine-run call above the ceiling its lane renders."""

    ceiling = None if policy is None else policy.call_target_ceiling(lane)
    if policy is None or ceiling is None:
        return
    requested = policy.target_count(arguments)
    if requested <= ceiling:
        return
    argument = policy.target_count_argument
    raise CapabilityInputError(
        "execution_not_supported_for_requested_shape",
        f"No query ran: {requested} {argument} exceed the per-call bound "
        f"of {ceiling}.",
        {"requested_targets": requested, "execution_limits": policy.projection()},
        recovery=ToolRecovery(
            RecoveryKind.REDUCE_SCOPE,
            f"Select at most {ceiling} {argument} per call.",
        ),
    )


# Drift found after approval is final for that request. Every post-approval
# ``state_changed`` result carries this one recovery, whatever the capability;
# staleness found before approval keeps the owning domain's re-preview advice.
APPROVED_TARGET_CHANGED = ToolRecovery(
    RecoveryKind.TERMINAL,
    "The approved target changed while approval was pending; nothing was "
    "written. Report the conflict and the target's current value to the user. "
    "Do not preview again or request approval again unless the user asks.",
)
_UNCERTAIN_OUTCOME = ToolRecovery(
    RecoveryKind.TERMINAL,
    "The outcome is unknown; report it and check the target before any "
    "further action. Never repeat the call.",
)
_CONTROL_TOOL_NAMES = RESERVED_TOOL_NAMES


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
                "and governance still apply. Pinned names are already callable and "
                "are skipped. Retains bounded contracts in the result; "
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

    @staticmethod
    def _catalog_owners(
        registry: CapabilityRegistry, domains: tuple[CapabilityDomain, ...]
    ) -> dict[str, CapabilityDomain]:
        """Validate a complete host-owned catalog before it is published."""
        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("registry must be CapabilityRegistry")
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
        return owners

    def _replace_catalog(
        self, registry: CapabilityRegistry, domains: tuple[CapabilityDomain, ...]
    ) -> None:
        """Publish between executions, under the host's system admission lease."""
        owners = self._catalog_owners(registry, domains)
        self._registry = registry
        self._domains = owners
        self._owner_management_locks: dict[str, asyncio.Lock] = {
            owner_id: self._owner_management_locks.get(owner_id, asyncio.Lock())
            for owner_id in owners
        }
        self._search_cursor_key = secrets.token_bytes(32)

    def __init__(
        self,
        registry: CapabilityRegistry,
        domains: tuple[CapabilityDomain, ...],
        *,
        approval_handler: ApprovalHandler | None = None,
        admission_coordinator: CapabilityAdmissionCoordinator | None = None,
        effect_coordinator: CapabilityEffectCoordinator | None = None,
        owner_management_locks: Mapping[str, asyncio.Lock] | None = None,
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
        domains = tuple(domains)
        owners = self._catalog_owners(registry, domains)
        if approval_handler is not None and not callable(approval_handler):
            raise TypeError("approval_handler must be callable or None")
        if admission_coordinator is not None and not all(
            callable(getattr(admission_coordinator, name, None))
            for name in ("source_resource_permit", "mcp_permit")
        ):
            raise TypeError("admission_coordinator is invalid")
        if effect_coordinator is not None and not callable(
            getattr(effect_coordinator, "acquire", None)
        ):
            raise TypeError("effect_coordinator is invalid")
        supplied_owner_locks = dict(owner_management_locks or {})
        if not set(supplied_owner_locks) <= set(owners) or any(
            not isinstance(lock, asyncio.Lock) for lock in supplied_owner_locks.values()
        ):
            raise TypeError("owner management locks are invalid")
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
        self._admission_coordinator = admission_coordinator
        self._effect_coordinator = effect_coordinator
        self._local_effect_lock = asyncio.Lock()
        self._owner_management_locks = {
            owner_id: supplied_owner_locks.get(owner_id, asyncio.Lock())
            for owner_id in owners
        }
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

    async def freeze_source_scope(self, run: RunInput) -> RunInput:
        """Return ``run`` with its effective source scope resolved before it starts.

        A machine run's transcript then records the same frozen scope its
        source and resource handles are numbered from, as a foreground run's
        does, so its executed calls stay derivable from persisted facts.
        Preparation re-resolves against this scope and can only narrow it.
        """

        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        if self._source_scope_resolver is None:
            return run
        return replace(
            run, resolved_source_scope=await self._source_scope_resolver(run)
        )

    async def prepare_run(self, run: RunInput) -> RunToolCatalog:
        """Prepare the complete immutable applicable catalog exactly once."""

        return await self._prepare_run(run, session=None)

    async def prepare_session(self, session: RunSession) -> RunToolCatalog:
        """Prepare tools from one exact immutable admitted session."""

        if not isinstance(session, RunSession):
            raise TypeError("session must be RunSession")
        return await self._prepare_run(session.run, session=session)

    async def _prepare_run(
        self,
        run: RunInput,
        *,
        session: RunSession | None,
    ) -> RunToolCatalog:
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
            # A domain may also withhold mode routes no current source can serve.
            project_routes = cast(
                Callable[..., Awaitable[Any]] | None,
                getattr(domain, "project_routes", None),
            )
            names, withheld = (
                await project_routes(run, session)
                if callable(project_routes)
                else (await _domain_project(domain, run, session), frozenset())
            )
            for name in names:
                if name in projected:
                    raise ValueError(f"tool projected by multiple domains: {name}")
                view = self._registry.resolve_tool(name)[0]
                admitted_by_capability = {
                    capability_id: self._admitted_catalog_entry(
                        run,
                        session,
                        name,
                        owner_id,
                        capability_id,
                    )
                    for capability_id in self._registry.tool_capability_ids(name)
                }
                if not view.mode_capability_ids:
                    admitted = admitted_by_capability[view.capability_id]
                    if admitted is not None:
                        projected[name] = admitted
                    continue
                routes = tuple(
                    (mode, route_entry)
                    for mode, capability_id in view.mode_capability_ids
                    if capability_id not in withheld
                    and (route_entry := admitted_by_capability[capability_id])
                    is not None
                )
                if not routes:
                    continue
                primary = (
                    None
                    if view.capability_id in withheld
                    else admitted_by_capability[view.capability_id]
                ) or routes[0][1]
                projected[name] = replace(primary, routes=routes)
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
        lane = call_target_lane(run.origin)
        pinned_entries = tuple(
            entry for entry in entries if entry.load_mode is ToolLoadMode.PINNED
        )
        pinned_definitions = tuple(
            sorted(
                (
                    _entry_definition(self._registry, entry, lane)
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
            call_target_lane=lane,
            source_scope=run.resolved_source_scope,
            session=session,
        )

    def _admitted_catalog_entry(
        self,
        run: RunInput,
        session: RunSession | None,
        name: str,
        owner_id: str,
        capability_id: str,
    ) -> RunToolCatalogEntry | None:
        view, capability, resolved_owner = self._registry.resolve_tool_owner(
            name, capability_id=capability_id
        )
        if resolved_owner != owner_id:
            raise ValueError(f"tool projected by the wrong domain: {name}")
        if run.execution_scope is not None and not run.execution_scope.allows(
            capability
        ):
            return None
        schema_digest = _sha256_digest(capability.input_schema)
        origin_digest = view.origin_revision_digest or _sha256_digest(
            {
                "domain_owner_id": owner_id,
                "capability_id": capability.id,
                "executor_id": capability.executor_id,
                "input_schema_digest": schema_digest,
            }
        )
        return RunToolCatalogEntry(
            view=view,
            capability=capability,
            domain_owner_id=owner_id,
            executor_id=capability.executor_id,
            input_schema_digest=schema_digest,
            origin_revision_digest=origin_digest,
            toolbox_id=view.presentation.toolbox_id,
            load_mode=view.presentation.load_mode,
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
            lane=catalog.call_target_lane,
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
                        _entry_definition(
                            self._registry, entry, catalog.call_target_lane
                        )
                        for entry in callable_entries
                    ),
                    *catalog.control_definitions,
                ),
                key=lambda item: item.name,
            )
        )
        _validate_step_surface(provider_definitions, self._limits)
        loaded_definition_bytes = _definition_bytes(
            tuple(
                _entry_definition(self._registry, entry, catalog.call_target_lane)
                for entry in loaded
            )
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
                call_target_lane=catalog.call_target_lane,
                source_scope=catalog.source_scope,
            ),
            activation_digest=activation_digest,
            provider_definitions=provider_definitions,
            catalog_entries=catalog.entries,
            callable_entries=callable_entries,
            loaded_entries=loaded,
            loaded_definition_bytes=loaded_definition_bytes,
            call_target_lane=catalog.call_target_lane,
            source_scope=catalog.source_scope,
            session=catalog.session,
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
        refs = _BATCH_RESULT_REFS.set(_batch_result_refs(messages, calls))
        try:
            return await self._execute_all(
                run,
                calls,
                projection=projection,
                messages=messages,
                sensitivity=sensitivity,
                session=None,
            )
        finally:
            _BATCH_RESULT_REFS.reset(refs)

    async def execute_session(
        self,
        session: RunSession,
        calls: tuple[ToolCall, ...],
        *,
        projection: object,
        messages: tuple[CanonicalMessage, ...],
        sensitivity: ModelSensitivity,
    ) -> ToolBatchOutcome:
        if not isinstance(session, RunSession):
            raise TypeError("session must be RunSession")
        refs = _BATCH_RESULT_REFS.set(_batch_result_refs(messages, calls))
        try:
            return await self._execute_all(
                session.run,
                calls,
                projection=projection,
                messages=messages,
                sensitivity=sensitivity,
                session=session,
            )
        finally:
            _BATCH_RESULT_REFS.reset(refs)

    async def _execute_all(
        self,
        run: RunInput,
        calls: tuple[ToolCall, ...],
        *,
        projection: object,
        messages: tuple[CanonicalMessage, ...],
        sensitivity: ModelSensitivity,
        session: RunSession | None,
    ) -> ToolBatchOutcome:
        if not isinstance(run, RunInput):
            raise TypeError("run must be RunInput")
        if not isinstance(sensitivity, ModelSensitivity):
            raise TypeError("tool batch sensitivity must be ModelSensitivity")
        if not isinstance(projection, StepToolProjection):
            raise TypeError("projection must be StepToolProjection")
        prepared_session = projection.session
        if session is None:
            if prepared_session is not None:
                raise ValueError("step projection belongs to an admitted run session")
        elif (
            prepared_session is None
            or prepared_session.run != session.run
            or prepared_session.writer is not session.writer
            or prepared_session.options != session.options
            or prepared_session.evidence is not session.evidence
        ):
            raise ValueError("step projection belongs to a different run session")
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
            lane=projection.call_target_lane,
        )
        if (
            projection.call_target_lane is not call_target_lane(run.origin)
            or projection.loaded_entries != loaded
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
        result_refs = tool_result_refs(
            messages, uncitable_tool_names=_CONTROL_TOOL_NAMES
        )
        if session is not None:
            analysis = self._domains.get("analysis")
            if analysis is not None:
                for resolved in resolved_calls:
                    if (
                        resolved.entry is not None
                        and resolved.entry.capability.id == "analysis.execute"
                    ):
                        cast("AnalysisCapabilityDomain", analysis).bind_parent(
                            session,
                            run,
                            resolved.target_call,
                            programmatic_reader(
                                self, session, run, resolved.target_call, projection
                            ),
                            sensitivity,
                            self._analysis_authority_reader(session, run),
                            self._limits.max_tool_calls_per_run,
                        )
        terminators = tuple(
            resolved
            for resolved in resolved_calls
            if resolved.entry is not None
            and resolved.entry.capability.machine_run_directive_kind is not None
        )
        if terminators and (len(terminators) != 1 or len(resolved_calls) != 1):
            return ToolBatchOutcome(
                tuple(
                    _bounded_tool_result(
                        resolved.outer_call,
                        _error(
                            resolved.outer_call,
                            "machine_termination_not_exclusive",
                            "A call that ends the run must be the only call in its step.",
                        ),
                        self._limits,
                    )
                    for resolved in resolved_calls
                )
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
                        result_refs=result_refs,
                        session=session,
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
                        session=session,
                        result_refs=result_refs,
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
        bounded = tuple(
            _bounded_tool_result(
                resolved.outer_call,
                result,
                self._limits,
            )
            for resolved, result in zip(resolved_calls, ordered, strict=True)
        )
        directive: MachineRunDirective | None = None
        if terminators and not bounded[0].is_error:
            terminator_entry = terminators[0].entry
            assert terminator_entry is not None
            kind = terminator_entry.capability.machine_run_directive_kind
            assert isinstance(kind, MachineRunDirectiveKind)
            directive = MachineRunDirective(kind=kind, tool_call_id=bounded[0].call_id)
        return ToolBatchOutcome(bounded, machine_run_directive=directive)

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
            # Unset nulls are dropped once, for every tool, before the mode is
            # chosen; the outer call keeps exactly what the model sent.
            outer = call
            call = replace(
                call,
                arguments=omit_unset_arguments(
                    call.arguments,
                    tuple(item.input_schema for item in entry.capabilities),
                ),
            )
        if entry is not None and entry.routes:
            # A mode-routed tool selects one admitted capability; the selected
            # capability then runs its own unchanged execution path.
            selector = entry.view.mode_argument
            mode = tool_call_mode(entry.view, call.arguments)
            routed = entry.route(mode)
            if routed is None:
                if mode in entry.view.presence_modes:
                    return _ResolvedCall(
                        outer,
                        call,
                        failure=_error(
                            call,
                            "invalid_argument_value",
                            "This form of the tool is not available in this run.",
                            recovery_details(
                                ToolRecovery(
                                    RecoveryKind.CORRECT_ARGUMENTS, f"Omit {mode}."
                                ),
                                {
                                    "name": mode,
                                    "constraint": "presence",
                                    "execution_disposition": "not_dispatched",
                                },
                            ),
                        ),
                    )
                if entry.view.mode_by_presence:
                    present = selector in call.arguments
                    return _ResolvedCall(
                        outer,
                        call,
                        failure=_error(
                            call,
                            "invalid_argument_value",
                            "This form of the tool is not available in this run.",
                            recovery_details(
                                ToolRecovery(
                                    RecoveryKind.CORRECT_ARGUMENTS,
                                    (
                                        f"Omit {selector}."
                                        if present
                                        else f"Provide {selector}."
                                    ),
                                ),
                                {
                                    "name": selector,
                                    "constraint": "presence",
                                    "execution_disposition": "not_dispatched",
                                },
                            ),
                        ),
                    )
                allowed = [
                    name
                    for name, _ in entry.routes
                    if name not in entry.view.presence_modes
                ]
                return _ResolvedCall(
                    outer,
                    call,
                    failure=_error(
                        call,
                        "invalid_argument_value",
                        f"Choose one {selector} this tool admits in this run.",
                        recovery_details(
                            ToolRecovery(
                                RecoveryKind.CORRECT_ARGUMENTS,
                                f"Set {selector} to one of: "
                                + ", ".join(allowed)
                                + ".",
                            ),
                            {
                                "name": selector,
                                "constraint": "enum",
                                "allowed": allowed,
                                "execution_disposition": "not_dispatched",
                            },
                        ),
                    ),
                )
            unknown = tool_route_unknown_arguments(
                entry.view, routed.capability, call.arguments
            )
            if unknown:
                if entry.view.mode_by_presence:
                    condition = (
                        f"with {selector}"
                        if selector in call.arguments
                        else f"without {selector}"
                    )
                elif mode in entry.view.presence_modes:
                    condition = f"with {mode}"
                else:
                    condition = f"when {selector}={mode}"
                return _ResolvedCall(
                    outer,
                    call,
                    failure=_error(
                        call,
                        "invalid_argument_for_mode",
                        f"{', '.join(unknown)} do not apply {condition}.",
                        recovery_details(
                            ToolRecovery(
                                RecoveryKind.CORRECT_ARGUMENTS,
                                f"Remove {', '.join(unknown)} or choose the "
                                f"{selector} they belong to.",
                            ),
                            {
                                "names": list(unknown),
                                "constraint": "route_arguments",
                                "execution_disposition": "not_dispatched",
                            },
                        ),
                    ),
                )
            target = replace(
                call,
                arguments=tool_route_arguments(
                    entry.view, routed.capability, call.arguments
                ),
            )
            return _ResolvedCall(outer, target, entry=routed)
        if entry is not None:
            return _ResolvedCall(outer, call, entry=entry)
        control_names = {item.name for item in projection.provider_definitions}
        if call.name not in _CONTROL_TOOL_NAMES or call.name not in control_names:
            loadable = "toolbox_load" in control_names and any(
                item.view.name == call.name and item.load_mode is ToolLoadMode.ON_DEMAND
                for item in projection.catalog_entries
            )
            return _ResolvedCall(
                call,
                call,
                failure=_error(
                    call,
                    "tool_not_available",
                    "The requested tool is not available for this step projection.",
                    recovery_details(
                        (
                            ToolRecovery(
                                RecoveryKind.CORRECT_ARGUMENTS,
                                f"Load {call.name} with toolbox_load, then call "
                                "it in the next step.",
                                tool_name="toolbox_load",
                            )
                            if loadable
                            else ToolRecovery(
                                RecoveryKind.CORRECT_ARGUMENTS,
                                "Call only a tool from the current tool list.",
                            )
                        ),
                        {
                            "tool_name": call.name,
                            "execution_disposition": "not_dispatched",
                        },
                    ),
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
        session: RunSession | None,
        result_refs: ToolResultRefs | None = None,
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
            if session is not None:
                session.evidence.record_outer_dispatch(resolved.outer_call.id)
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
            session=session,
            projection=projection,
            result_refs=result_refs,
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
        if control_name == "toolbox_search" or (
            control_name == "toolbox_load"
            and prepared_result is not None
            and not prepared_result.is_error
        ):
            result = await self._applicable_control_result(
                run, call, control_name, projection, sensitivity, prepared_result
            )
        elif prepared_result is not None:
            result = prepared_result
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

    async def _current_tool_names(
        self,
        run: RunInput,
        projection: StepToolProjection,
    ) -> frozenset[str] | None:
        """Intersect frozen candidates with current local domain applicability."""

        names: set[str] = set()
        try:
            for owner_id in sorted(
                {entry.domain_owner_id for entry in projection.catalog_entries}
            ):
                names.update(
                    await _domain_project(
                        self._domains[owner_id], run, projection.session
                    )
                )
        except Exception:
            return None
        return frozenset(
            entry.view.name
            for entry in projection.catalog_entries
            if entry.view.name in names
        )

    async def _applicable_control_result(
        self,
        run: RunInput,
        call: ToolCall,
        control_name: str,
        projection: StepToolProjection,
        sensitivity: ModelSensitivity,
        prepared_result: ToolResultBlock | None,
    ) -> ToolResultBlock:
        current = await self._current_tool_names(run, projection)
        if current is None:
            return _error(
                call,
                f"{control_name}_unavailable",
                "Current local tool admission could not be checked; no tools "
                "were loaded or reported.",
                recovery_details(
                    ToolRecovery(
                        RecoveryKind.TERMINAL,
                        "Report the blocker; tool availability is unknown.",
                    )
                ),
            )
        if control_name == "toolbox_search":
            return self._toolbox_search(
                call, projection, sensitivity=sensitivity, current_names=current
            )
        assert prepared_result is not None
        requested = cast(tuple[str, ...], call.arguments["tool_names"])
        unavailable = [name for name in requested if name not in current]
        if unavailable:
            return _error(
                call,
                "toolbox_tool_not_available",
                "A requested tool is not currently applicable in this run, so "
                "nothing was loaded: " + ", ".join(unavailable) + ".",
                recovery_details(
                    ToolRecovery(
                        RecoveryKind.TERMINAL,
                        "These tools cannot be called in this run; continue with "
                        "the applicable tools or report the limitation.",
                    ),
                    {"tool_names": unavailable},
                ),
            )
        return prepared_result

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
        current_names: frozenset[str],
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
        # An exact registered tool name gets a definite availability answer,
        # so a missing tool is not mistaken for a ranking miss.
        exact_name = (
            query.strip() if query.strip() in self._registry.tool_names else None
        )
        loaded_names = {entry.view.name for entry in projection.loaded_entries}
        scored: list[tuple[int, str, str, RunToolCatalogEntry]] = []
        for entry in projection.catalog_entries:
            if entry.view.name not in current_names:
                continue
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
                **(
                    {}
                    if exact_name is None
                    else {
                        "exact_tool_name": exact_name,
                        "exact_tool_available": exact_name in current_names,
                    }
                ),
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
        already_pinned: list[str] = []
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
                already_pinned.append(name)
                continue
            if not _entry_resolves_exactly(entry, self._registry):
                return _error(
                    call,
                    "toolbox_load_stale",
                    "A requested tool no longer resolves to its frozen run contract.",
                    {"tool_name": name},
                )
            requested.append(entry)
        if not requested:
            return _error(
                call,
                "toolbox_load_invalid",
                "Pinned tools are already callable; load only on-demand tools.",
                {"already_pinned_names": sorted(already_pinned)},
            )
        loaded = tuple(sorted(requested, key=lambda item: item.view.name))
        definitions = tuple(
            _entry_definition(self._registry, entry, projection.call_target_lane)
            for entry in loaded
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
                        _entry_definition(
                            self._registry, item, projection.call_target_lane
                        )
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
        if already_pinned:
            data["already_pinned_names"] = sorted(already_pinned)
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

    def _analysis_authority_reader(self, session: RunSession, run: RunInput):
        async def revalidate(call: ToolCall, sensitivity: ModelSensitivity) -> None:
            view, capability, owner = self._registry.resolve_tool_owner(
                call.name,
                capability_id=self._registry.call_capability(
                    call.name, call.arguments
                ).id,
            )
            if not (
                self._analysis_read_eligible(owner, capability)
                or self._analysis_capture_eligible(owner, capability)
            ):
                raise CapabilityInputError(
                    "analysis_authority_revoked",
                    "An admitted input is outside the analytical read surface.",
                )
            domain = self._domains[owner]
            if view.name not in await _domain_project(domain, run, session, call=call):
                raise CapabilityInputError(
                    "analysis_authority_revoked",
                    "An admitted input is no longer readable.",
                )
            arguments = await self._registry.validate_arguments_async(
                capability.id,
                domain.normalize_arguments(
                    capability,
                    self._registry.target_arguments(
                        call.name,
                        self._registry.resolve_call(
                            call,
                            capability=capability,
                            result_refs=None,
                            source_scope=run.resolved_source_scope,
                        ).arguments,
                    ),
                ),
            )
            await _domain_prepare_call(
                domain,
                run,
                call,
                capability,
                arguments,
                request_sensitivity=sensitivity,
                session=session,
            )

        return revalidate

    async def handoff_analysis_recovery(self) -> None:
        analysis = self._domains.get("analysis")
        if analysis is not None:
            await cast("AnalysisCapabilityDomain", analysis).handoff_recovery()

    def _analysis_read_eligible(self, owner: str, capability: Capability) -> bool:
        local_artifact_read = owner == "artifacts" and capability.id in getattr(
            self._domains[owner], "programmatic_read_capability_ids", frozenset()
        )
        return (
            owner in {"data", "artifacts", "mcp"}
            and (
                capability.access_mode is AccessMode.READ
                or local_artifact_read
                and capability.access_mode is AccessMode.NONE
            )
            and capability.operational_effect is OperationalEffect.NONE
            and capability.artifact_policy is None
            and capability.machine_run_directive_kind is None
        )

    def _analysis_capture_eligible(self, owner: str, capability: Capability) -> bool:
        policy = capability.artifact_policy
        return (
            owner == "data"
            and capability.access_mode is AccessMode.READ
            and capability.operational_effect is OperationalEffect.NONE
            and capability.machine_run_directive_kind is None
            and policy is not None
            and "application/vnd.apache.arrow.file" in policy.allowed_media_types
            and policy.max_artifact_count == 1
        )

    async def _execute_one(
        self,
        run: RunInput,
        call: ToolCall,
        entry: RunToolCatalogEntry,
        *,
        sensitivity: ModelSensitivity,
        validated_arguments: FrozenJsonObject | None,
        session: RunSession | None,
        dispatch_hook: Callable[[], Awaitable[None]] | None = None,
        projection: StepToolProjection | None = None,
        result_refs: ToolResultRefs | None = None,
    ) -> ToolResultBlock:
        started = (
            asyncio.get_running_loop().time() if self._observer is not None else None
        )
        capability = entry.capability
        dispatch = _DispatchState()

        async def mark_dispatch() -> None:
            dispatch.started = True
            if dispatch_hook is not None:
                await dispatch_hook()
            elif session is not None:
                session.evidence.record_outer_dispatch(call.id)

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
            view, resolved, owner_id = self._registry.resolve_tool_owner(
                call.name, capability_id=entry.capability.id
            )
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
                # The same resolver derives every reader's executed record
                # (CapabilityRegistry.executed_tool_pairs) from the transcript.
                raw_arguments = domain.normalize_arguments(
                    capability,
                    self._registry.resolve_call(
                        call,
                        capability=capability,
                        result_refs=result_refs,
                        source_scope=run.resolved_source_scope,
                    ).arguments,
                )
                arguments = await self._registry.validate_arguments_async(
                    capability.id,
                    raw_arguments,
                )
            else:
                arguments = validated_arguments
            preference = (
                ExecutionPreference.AUTO
                if session is None
                else session.options.execution_preference
            )
            if run.origin is RunOrigin.USER:
                policy = capability.execution_admission_policy
                if policy is not None:
                    admission_error = policy.admission_error(arguments, preference)
                    if admission_error is not None:
                        raise await self._admission_failure(
                            run,
                            session,
                            projection,
                            policy,
                            arguments,
                            admission_error,
                        )
                elif (
                    preference is ExecutionPreference.DURABLE
                    and not capability.durable_foreground_entrypoint
                ):
                    raise CapabilityInputError(
                        "execution_not_supported_for_requested_shape",
                        "This capability is direct-only and cannot run as a job.",
                        recovery=ToolRecovery(
                            RecoveryKind.TERMINAL,
                            "This tool runs only in the foreground; durable "
                            "execution is not available for it.",
                        ),
                    )
            else:
                _require_call_target_ceiling(
                    capability.execution_admission_policy,
                    call_target_lane(run.origin),
                    arguments,
                )
            arguments = await _domain_prepare_call(
                domain,
                run,
                call,
                capability,
                arguments,
                request_sensitivity=sensitivity,
                session=session,
            )
            provenance = await _domain_call_provenance(
                domain, run, call, capability, arguments, session=session
            )
            resolved_capability, executor = self._registry.resolve_execution(
                capability.id
            )
            if (
                resolved_capability != capability
                or capability.id not in self._registry.tool_capability_ids(view.name)
                or self._registry.resolve_domain_owner(capability.id) != owner_id
            ):
                raise ValueError("tool execution identity changed")
            execution = ToolExecution(
                run_id=run.id,
                call_id=call.id,
                capability_id=capability.id,
                caller_principal_id=run.caller_principal_id,
                caller_principal_verified=run.caller_principal_verified,
                execution_scope=run.execution_scope,
                arguments=arguments,
                conversation_id=run.conversation_id or run.id,
                source_scope=run.resolved_source_scope,
                request_sensitivity=sensitivity,
                one_time_artifact_destinations=(
                    ()
                    if session is None
                    else session.options.one_time_artifact_destinations
                ),
                provenance=provenance,
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
                    session=session,
                    dispatch=dispatch,
                )
            else:
                output, artifact_refs = await self._execute_no_effect(
                    run,
                    call,
                    capability,
                    executor,
                    arguments,
                    domain,
                    sensitivity=sensitivity,
                    session=session,
                    catalog_entry=entry,
                    dispatch_hook=mark_dispatch,
                    provenance=provenance,
                )
                result = _classified_success(call, output, artifact_refs=artifact_refs)
        except _ToolExecutionInterrupted:
            raise
        except asyncio.CancelledError:
            raise
        except BaseException as error:
            result = self._exception_result(call, error, domain)
        if result.is_error and not dispatch.started:
            # Proven by control flow: the executor or effect was never reached.
            result = _with_execution_disposition(result, "not_dispatched")
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

    async def _durable_entrypoint(
        self,
        run: RunInput,
        session: RunSession | None,
        projection: StepToolProjection | None,
    ) -> tuple[RunToolCatalogEntry, bool] | None:
        """Return the durable entrypoint this run can call, and if it is loaded.

        Availability is capability metadata intersected with the prepared
        candidates and the owning domain's current local projection.
        """

        if projection is None:
            return None
        for entry in projection.catalog_entries:
            if not entry.capability.durable_foreground_entrypoint:
                continue
            if not _entry_resolves_exactly(entry, self._registry):
                continue
            try:
                names = await _domain_project(
                    self._domains[entry.domain_owner_id], run, session
                )
            except Exception:
                continue
            if entry.view.name in names:
                return entry, entry in projection.callable_entries
        return None

    async def _admission_failure(
        self,
        run: RunInput,
        session: RunSession | None,
        projection: StepToolProjection | None,
        policy: ExecutionAdmissionPolicy,
        arguments: Mapping[str, object],
        admission_error: str,
    ) -> CapabilityInputError:
        requested = policy.target_count(arguments)
        argument = policy.target_count_argument or "targets"
        durable = (
            await self._durable_entrypoint(run, session, projection)
            if admission_error == "durable_execution_required"
            else None
        )
        if requested < 1:
            recovery = ToolRecovery(
                RecoveryKind.CORRECT_ARGUMENTS,
                f"Select at least one entry in {argument}.",
            )
        elif durable is not None:
            entry, callable_now = durable
            recovery = ToolRecovery(
                RecoveryKind.REDUCE_SCOPE,
                f"Select at most {policy.inline_max_targets} {argument} per call, "
                f"or submit up to {policy.graph_max_targets} as one durable task "
                f"with {entry.view.name}"
                + ("" if callable_now else " after loading it with toolbox_load")
                + ".",
                tool_name=entry.view.name,
            )
        elif policy.inline_eligible and requested > policy.inline_max_targets:
            recovery = ToolRecovery(
                RecoveryKind.REDUCE_SCOPE,
                f"Select at most {policy.inline_max_targets} {argument} per call.",
            )
        else:
            recovery = ToolRecovery(
                RecoveryKind.TERMINAL,
                "No execution path in this run admits this shape; report the "
                "limitation.",
            )
        if policy.inline_eligible and requested > policy.inline_max_targets:
            message = (
                f"No query ran: {requested} {argument} exceed the per-call bound "
                f"of {policy.inline_max_targets}."
            )
        else:
            message = "No query ran: this shape is not admitted in the foreground."
        return CapabilityInputError(
            admission_error,
            message,
            {
                "requested_targets": requested,
                "execution_limits": policy.projection(),
                "durable_available": durable is not None,
            },
            recovery=recovery,
        )

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
        session: RunSession | None = None,
        catalog_entry: RunToolCatalogEntry | None = None,
        dispatch_hook: Callable[[], Awaitable[None]] | None = None,
        provenance: CallProvenance = CallProvenance(),
    ) -> tuple[ToolOutput, tuple[ArtifactRef, ...]]:
        if capability.operational_effect is not OperationalEffect.NONE:
            raise ValueError("non-effect execution requires operational effect none")
        execution = ToolExecution(
            run_id=run.id,
            call_id=call.id,
            capability_id=capability.id,
            caller_principal_id=run.caller_principal_id,
            caller_principal_verified=run.caller_principal_verified,
            execution_scope=run.execution_scope,
            arguments=arguments,
            conversation_id=run.conversation_id or run.id,
            source_scope=run.resolved_source_scope,
            request_sensitivity=sensitivity,
            one_time_artifact_destinations=(
                ()
                if session is None
                else session.options.one_time_artifact_destinations
            ),
            provenance=provenance,
        )
        permit = await self._call_io_permit(
            run,
            capability,
            arguments,
            catalog_entry,
            session=session,
        )
        if permit is None:
            if dispatch_hook is not None:
                await dispatch_hook()
            elif session is not None:
                session.evidence.record_outer_dispatch(call.id)
            candidate = await executor.execute(execution)
        else:
            async with permit:
                if dispatch_hook is not None:
                    await dispatch_hook()
                elif session is not None:
                    session.evidence.record_outer_dispatch(call.id)
                candidate = await executor.execute(execution)
        if not isinstance(candidate, ToolOutput):
            raise ToolOutputValidationError("executor did not return ToolOutput")
        output = await _domain_finalize_output(
            domain,
            run,
            call,
            capability,
            arguments,
            candidate,
            request_sensitivity=sensitivity,
            session=session,
        )
        output = self._registry.validate_output(capability.id, output)
        _validate_output_execution_scope(run, capability, output)
        drafts = (
            () if output.artifact is None else (output.artifact,)
        ) + output.additional_artifacts
        policy = capability.artifact_policy
        if drafts:
            if policy is None or len(drafts) > policy.max_artifact_count:
                raise ToolOutputValidationError(
                    "artifact drafts exceed the capability count"
                )
            if (
                sum(len(draft.content) for draft in drafts)
                > policy.max_total_bytes_per_call
            ):
                raise ToolOutputValidationError(
                    "artifact drafts exceed the per-call byte allowance"
                )
            for draft in drafts:
                self._validate_artifact_draft(capability, draft)
        refs: list[ArtifactRef] = []
        try:
            if not drafts:
                await self._commit_artifact_output(run, call, capability, output)
            for draft in drafts:
                ref = await self._commit_artifact_output(
                    run,
                    call,
                    capability,
                    replace(output, artifact=draft, additional_artifacts=()),
                )
                assert ref is not None
                refs.append(ref)
                if capability.id == "analysis.execute":
                    cast("AnalysisCapabilityDomain", domain).artifact_committed(
                        run.id, ref.byte_size
                    )
        except ArtifactError as error:
            if capability.id != "analysis.execute":
                raise
            output = cast("AnalysisCapabilityDomain", domain).artifact_save_failed(
                output, error.code
            )
        return output, tuple(refs)

    async def _call_io_permit(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
        catalog_entry: RunToolCatalogEntry | None,
        *,
        session: RunSession | None,
    ) -> AdmissionPermit | None:
        coordinator = self._admission_coordinator
        if coordinator is None:
            return None
        # The interpreter performs no source I/O. Its brokered children acquire
        # their own exact permits; holding a source permit here would deadlock a
        # scheduled cell that reads its sole admitted source.
        if capability.id == "analysis.execute":
            return None
        deadline = None if session is None else session.deadline
        cancellation = None if session is None else session.cancellation
        presentation = (
            None if catalog_entry is None else catalog_entry.view.connector_presentation
        )
        if isinstance(presentation, Mapping):
            binding_id = presentation.get("id")
            if isinstance(binding_id, str) and binding_id:
                return await coordinator.mcp_permit(
                    f"mcp:{run.agent_id}:{binding_id}",
                    deadline=deadline,
                    cancellation=cancellation,
                )
        source_id = arguments.get("source_id")
        if isinstance(source_id, str) and source_id:
            return await coordinator.source_resource_permit(
                f"source:{source_id}",
                deadline=deadline,
                cancellation=cancellation,
            )
        inputs = arguments.get("inputs")
        if isinstance(inputs, tuple) and inputs:
            # Each distinct input source is held once, in sorted order, so two
            # inputs on one capacity-one source and concurrent multi-source
            # calls cannot deadlock.
            permits = AsyncExitStack()
            try:
                for input_source_id in sorted(
                    {
                        cast(str, cast(Mapping[str, object], item)["source_id"])
                        for item in inputs
                    }
                ):
                    await permits.enter_async_context(
                        await coordinator.source_resource_permit(
                            f"source:{input_source_id}",
                            deadline=deadline,
                            cancellation=cancellation,
                        )
                    )
            except BaseException:
                await permits.aclose()
                raise
            return permits
        if (
            capability.access_mode is AccessMode.READ
            and run.execution_scope is not None
            and len(run.execution_scope.allowed_source_ids) == 1
        ):
            return await coordinator.source_resource_permit(
                f"source:{run.execution_scope.allowed_source_ids[0]}",
                deadline=deadline,
                cancellation=cancellation,
            )
        return None

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
        session: RunSession | None,
        dispatch: _DispatchState | None = None,
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
        plan = await _domain_side_effect_plan(
            domain,
            run,
            call,
            capability,
            execution,
            fingerprint,
            session=session,
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
                        recovery_details(
                            ToolRecovery(
                                RecoveryKind.REQUIRES_HUMAN_PERMISSION,
                                "A person must approve this action; no approval "
                                "path exists in this run.",
                            ),
                            {"capability_id": capability.id},
                        ),
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
                requester=None if session is None else session.options.requester,
            )
            self._emit_approval_requested(
                run,
                call,
                capability,
                catalog_entry=catalog_entry,
            )
            try:
                with nullcontext() if session is None else session.approval_wait():
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
                        recovery_details(
                            ToolRecovery(
                                RecoveryKind.REQUIRES_HUMAN_PERMISSION,
                                "The person declined this exact action; do not "
                                "repeat it unless they ask again.",
                            ),
                            {"capability_id": capability.id},
                        ),
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
            catalog_entry=catalog_entry,
            session=session,
            dispatch=dispatch,
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
        catalog_entry: RunToolCatalogEntry,
        session: RunSession | None,
        dispatch: _DispatchState | None = None,
    ) -> tuple[
        ToolResultBlock,
        ToolBatchInterruption | None,
        ToolBatchCertainty,
    ]:
        async with self._side_effect_lane(
            capability,
            session,
            run=run,
            arguments=arguments,
            catalog_entry=catalog_entry,
        ):
            await self._validate_execution_contracts(run)
            if capability.effect_receipt_policy is not None:
                current_arguments = await _domain_prepare_call(
                    domain,
                    run,
                    call,
                    capability,
                    arguments,
                    request_sensitivity=sensitivity,
                    session=session,
                )
                current_provenance = await _domain_call_provenance(
                    domain,
                    run,
                    call,
                    capability,
                    current_arguments,
                    session=session,
                )
                if (
                    current_arguments != arguments
                    or current_provenance != execution.provenance
                ):
                    raise CapabilityInputError(
                        "state_changed",
                        "Current effect binding changed after approval.",
                        recovery=APPROVED_TARGET_CHANGED,
                    )
            if plan.recheck_after_approval:
                try:
                    current = await side_effect.preflight(execution)
                    if not isinstance(current, FrozenJsonObject):
                        raise ValueError(
                            "side-effect preflight must return FrozenJsonObject"
                        )
                    current_plan = await _domain_side_effect_plan(
                        domain,
                        run,
                        call,
                        capability,
                        execution,
                        current,
                        session=session,
                    )
                    if (
                        capability.effect_receipt_policy is not None
                        and current_plan != plan
                    ):
                        raise CapabilityInputError(
                            "state_changed",
                            "Current effect authorization changed after approval.",
                            recovery=APPROVED_TARGET_CHANGED,
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
                            recovery_details(
                                APPROVED_TARGET_CHANGED,
                                {"capability_id": capability.id},
                            ),
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
                            recovery_details(
                                APPROVED_TARGET_CHANGED,
                                {"capability_id": capability.id},
                            ),
                        ),
                        None,
                        ToolBatchCertainty.DEFINITE,
                    )
            execution = replace(execution, approved_preflight=fingerprint)
            if capability.effect_receipt_policy is not None:
                if dispatch is not None:
                    # Reservation starts the receipt-owned effect boundary.
                    dispatch.started = True
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
                    session=session,
                )
            if session is not None:
                session.evidence.record_outer_dispatch(call.id)
            if dispatch is not None:
                dispatch.started = True
            (
                candidate,
                execution_error,
                interruption_kind,
                outcome_certainty,
            ) = await _execute_definitely(
                side_effect,
                execution,
                recovery_timeout_seconds=(self._side_effect_recovery_timeout_seconds),
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
            output = await _domain_finalize_output(
                domain,
                run,
                call,
                capability,
                arguments,
                output,
                request_sensitivity=sensitivity,
                session=session,
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

    @asynccontextmanager
    async def _side_effect_lane(
        self,
        capability: Capability,
        session: RunSession | None,
        *,
        run: RunInput,
        arguments: Mapping[str, object],
        catalog_entry: RunToolCatalogEntry,
    ) -> AsyncIterator[None]:
        if capability.effect_receipt_policy is None:
            owner_id = self._registry.resolve_domain_owner(capability.id)
            async with self._owner_management_locks[owner_id]:
                yield
            return
        if self._effect_coordinator is None:
            async with self._local_effect_lock:
                yield
            return
        permit = await self._effect_coordinator.acquire(
            deadline=None if session is None else session.deadline,
            cancellation=None if session is None else session.cancellation,
        )
        async with permit:
            conflict_permit = await self._effect_conflict_permit(
                run,
                capability,
                arguments,
                catalog_entry,
                session=session,
            )
            if conflict_permit is None:
                yield
            else:
                async with conflict_permit:
                    yield

    async def _effect_conflict_permit(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
        catalog_entry: RunToolCatalogEntry,
        *,
        session: RunSession | None,
    ) -> AdmissionPermit | None:
        coordinator = self._admission_coordinator
        if coordinator is None or capability.effect_receipt_policy is None:
            return None
        deadline = None if session is None else session.deadline
        cancellation = None if session is None else session.cancellation
        presentation = catalog_entry.view.connector_presentation
        if isinstance(presentation, Mapping):
            binding_id = presentation.get("id")
            if isinstance(binding_id, str) and binding_id:
                return await coordinator.mcp_permit(
                    f"mcp:{run.agent_id}:{binding_id}",
                    deadline=deadline,
                    cancellation=cancellation,
                )
        source_id = arguments.get("source_id")
        resource_id = arguments.get("resource_id")
        if isinstance(source_id, str) and isinstance(resource_id, str):
            return await coordinator.source_resource_permit(
                f"relwrite:{source_id}:{resource_id}",
                deadline=deadline,
                cancellation=cancellation,
            )
        return None

    def _validate_effect_plan(
        self,
        run: RunInput,
        capability: Capability,
        domain: CapabilityDomain,
        plan: SideEffectPlan,
    ) -> None:
        del run, capability, domain
        if plan.effect_intent is None:
            raise ValueError(
                "receipt-bearing effects require a canonical domain intent"
            )
        # Every run, foreground or machine, asks a person at the time (invariant 5).
        if not plan.approval_required:
            raise ValueError("external effects require exact per-call approval")

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
        session: RunSession | None,
    ) -> tuple[ToolResultBlock, ToolBatchInterruption | None, ToolBatchCertainty]:
        from .storage.errors import StorageError
        from .storage.sqlite_records import EffectReceipt, effect_receipt_id

        store = self._effect_receipts
        policy = capability.effect_receipt_policy
        assert store is not None and policy is not None
        self._validate_effect_plan(run, capability, domain, plan)
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
                "graph_task_binding": (
                    None
                    if scope is None or scope.graph_task_binding is None
                    else scope.graph_task_binding.digest
                ),
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
        )
        active_task = asyncio.current_task()
        cancellations_before_reservation = (
            0 if active_task is None else active_task.cancelling()
        )
        await store.start_effect_receipt(
            receipt,
            max_receipts_per_run=min(self._limits.max_tool_calls_per_run, 256),
        )
        execution = replace(execution, effect_receipt_id=receipt.receipt_id)
        observation: EffectObservation | None = None
        interruption: ToolBatchInterruption | None = None
        certainty = ToolBatchCertainty.DEFINITE
        # Storage failures still reach the execution owner, but only after the
        # reserved receipt is finished so it cannot stay STARTED in-process.
        storage_error: StorageError | None = None
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
                recovery_details(
                    ToolRecovery(
                        RecoveryKind.TERMINAL,
                        "The run was cancelled; its reserved receipt records that "
                        "nothing was applied.",
                    ),
                    {"execution_disposition": "not_dispatched"},
                ),
            )
        else:
            try:
                if session is not None:
                    session.evidence.record_outer_dispatch(call.id)
                (
                    candidate,
                    execution_error,
                    interruption,
                    certainty,
                ) = await _execute_definitely(
                    executor,
                    execution,
                    recovery_timeout_seconds=self._side_effect_recovery_timeout_seconds,
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
                    output = await _domain_finalize_output(
                        domain,
                        run,
                        call,
                        capability,
                        arguments,
                        candidate,
                        request_sensitivity=sensitivity,
                        session=session,
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
                try:
                    result = self._exception_result(call, error, domain)
                except StorageError as captured:
                    storage_error = captured
                    result = _error(
                        call,
                        "effect_receipt_unavailable",
                        "Storage failed after the effect was dispatched.",
                    )
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
                recovery_details(
                    _UNCERTAIN_OUTCOME, {"execution_disposition": "unknown"}
                ),
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
            if storage_error is not None:
                raise storage_error from error
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
        if storage_error is not None:
            raise storage_error
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
        self._validate_artifact_draft(capability, draft)
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
            caller_principal_id=run.caller_principal_id,
        )

    @staticmethod
    def _validate_artifact_draft(capability: Capability, draft: ArtifactDraft) -> None:
        policy = capability.artifact_policy
        if policy is None or policy.max_artifact_count < 1:
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

    def _exception_result(
        self,
        call: ToolCall,
        error: BaseException,
        domain: CapabilityDomain | None,
    ) -> ToolResultBlock:
        from .storage.errors import StorageError
        from .storage.sqlite_records import (
            EffectReceiptConflictError,
            EffectUnresolvedError,
        )

        if isinstance(error, StorageError):
            # Storage uncertainty belongs to the execution owner, never to a
            # model correction/retry loop. Preserve the typed failure.
            raise error
        if isinstance(error, EffectUnresolvedError):
            return _error(
                call,
                "effect_unresolved",
                str(error),
                recovery_details(
                    ToolRecovery(
                        RecoveryKind.TERMINAL,
                        "An earlier effect has no confirmed outcome; a person "
                        "must resolve its receipt before new effects run.",
                    ),
                    {
                        "receipt_ids": error.receipt_ids,
                        "omitted_count": error.omitted_count,
                    },
                ),
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
                recovery_details(
                    _UNCERTAIN_OUTCOME,
                    {
                        "execution_state": "started",
                        "execution_disposition": "unknown",
                        "outcome_certainty": ToolBatchCertainty.OUTCOME_UNKNOWN.value,
                    },
                ),
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
        run_origin: RunOrigin | None = None,
    ) -> None:
        data: dict[str, object] = {"call_id": call.id, "tool_name": call.name}
        if capability is not None:
            data["capability_id"] = capability.id
        if catalog_entry is not None:
            data.update(_toolbox_observation(catalog_entry))
        self._emit(AgentEventKind.TOOL_STARTED, run, data, run_origin=run_origin)

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
        run_origin: RunOrigin | None = None,
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
        try:
            data["result_bytes"] = _result_bytes(result)
        except Exception:
            pass
        if capability is not None:
            data["capability_id"] = capability.id
        if catalog_entry is not None:
            data.update(_toolbox_observation(catalog_entry))
        self._emit(
            AgentEventKind.TOOL_COMPLETED,
            run,
            data,
            run_origin=run_origin,
        )

    def _emit(
        self,
        kind: AgentEventKind,
        run: RunInput,
        data: Mapping[str, object],
        *,
        run_origin: RunOrigin | None = None,
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
                run_origin=(run_origin or run.origin).value,
            )
        except Exception:
            return
        _emit_safely(self._observer, event)


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
        call_target_lane=projection.call_target_lane,
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
        _entry_definition(registry, entry, projection.call_target_lane)
        for entry in loaded
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
                    _entry_definition(registry, entry, projection.call_target_lane)
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


__all__ = [
    "CapabilityDomain",
    "CapabilityFailure",
    "CapabilityRuntime",
    "MachineRunScope",
    "MachineScopeBuilder",
    "RunToolCatalog",
    "RunToolCatalogEntry",
    "SideEffectPlan",
    "StepToolProjection",
    "ToolboxManifestEntry",
    "machine_run_capability_ids",
    "machine_run_connector_binding_ids",
]
