"""Compose and own the in-process runtime for one persistent agent home."""

from __future__ import annotations

import asyncio
import hmac
import re
import secrets
import sqlite3
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Self, TypedDict, TypeVar, cast
from uuid import uuid4

if TYPE_CHECKING:
    from ..adapters.mcp import MCPServerBinding, MCPToolBinding

from .._json import FrozenJsonObject
from ..adapters.folder import SessionFolderSource
from ..adapters.local_file_query import LocalFileQueryBackend, LocalFileQueryLimits
from ..adapters.local_workspace import LocalWorkspaceBackend
from ..adapters.mcp import (
    MCPAdmissionError,
    MCPAuthentication,
    MCPAuthenticationError,
    MCPAuthenticationMode,
    MCPBindingState,
    MCPBindingStatus,
    MCPClientFactory,
    MCPConnectionProvider,
    MCPPersonalConnectionClient,
    MCPServerInspection,
    MCPToolSelection,
    SDKMCPClientFactory,
    check_personal_connection,
    mcp_binding_drift_reason,
    mcp_binding_from_inspection,
    mcp_execution_origin_digest,
)
from ..adapters.models import DiscoveryRequest, SourceRegistration
from ..adapters.postgresql import (
    POSTGRESQL_ADAPTER_ID,
    PostgreSQLProbeResult,
    PostgreSQLSource,
)
from ..adapters.postgresql_query import PostgreSQLReadDiagnostics
from ..adapters.postgresql_write import RelationalUpdateReadiness
from ..adapters.protocols import ResourceAdapter, ResourceAdapterError, ResourceSource
from ..adapters.registry import (
    SourceBackendContext,
    SourceOperation,
    SourceType,
    SourceTypeRegistry,
)
from ..adapters.sqlite import SQLiteSource
from ..artifacts.bytes import ArtifactByteStorage
from ..artifacts.delivery import (
    ArtifactDestinationGrant,
    LocalArtifactDelivery,
)
from ..artifacts.models import (
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactPayload,
    ArtifactRef,
    ArtifactState,
)
from ..artifacts.store import AgentHomeArtifactStore
from ..capabilities import (
    ApprovalDecision,
    ApprovalHandler,
    ApprovalRequest,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    CapabilityRegistry,
    EffectOutcome,
    ExecutionContractBindings,
    ExecutionContractReader,
    ToolView,
)
from ..capability_runtime import (
    CapabilityRuntime,
    MachineRunScope,
    machine_run_capability_ids,
    machine_run_connector_binding_ids,
)
from ..catalog.capabilities import (
    catalog_declarations,
)
from ..catalog.models import (
    CatalogLink,
    CatalogLinkStatus,
    CatalogResource,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSummary,
    CatalogSync,
    CatalogSyncStatus,
    RelationshipFieldPair,
)
from ..catalog.service import CatalogService
from ..config import (
    AgentConfig,
    AnalysisLimits,
    HostCapacityLimits,
    PostgreSQLReadLimits,
)
from ..context import AgentContextBuilder, _project_completed_history
from ..distribution import (
    DISTRIBUTION_DOMAIN_OWNER_ID,
    DeliveryInspection,
    DeliverySubjectKind,
    DeliveryView,
    DistributionCapabilityDomain,
    DistributionDestination,
    DistributionOwner,
    OutcomeContract,
    distribution_capability_declarations,
)
from ..domains.analysis import AnalysisCapabilityDomain
from ..domains.data import (
    ARTIFACT_DOMAIN_OWNER_ID,
    DATA_QUERY_MAX_BYTES,
    DATA_QUERY_MAX_ROWS,
    LOCAL_ARTIFACT_EDIT_CAPABILITY_IDS,
    LOCAL_ARTIFACT_EDIT_EXECUTOR_IDS,
    PATH_QUERY_CAPABILITY_ID,
    PATH_QUERY_EXECUTOR_ID,
    SOURCE_FILE_CAPABILITY_IDS,
    ArtifactCapabilityDomain,
    CatalogDataView,
    DataCapabilityDomain,
    artifact_declarations,
    data_query_declarations,
    exact_export_declarations,
    exact_export_outputs,
    relational_update_declarations,
    relational_update_preview_declarations,
    relational_upsert_declarations,
    resource_revision_observation_declarations,
    source_file_declarations,
)
from ..domains.data.catalog import RELATIONAL_WRITE_OPERATIONS
from ..domains.data.controller import DATA_DOMAIN_OWNER_ID
from ..domains.data.profile_jobs import (
    DATA_PROFILE_DOMAIN_OWNER_ID,
    DataProfileAdmission,
    DataProfileCapabilityDomain,
    data_profile_declarations,
)
from ..domains.mcp import (
    MCP_DOMAIN_OWNER_ID,
    MCPActivatedBinding,
    activate_mcp_domain,
    validate_mcp_model_compatibility,
)
from ..identity import AgentIdentity
from ..jobs.capabilities import (
    JOB_DOMAIN_OWNER_ID,
    JobCapabilityDomain,
    job_capability_declarations,
)
from ..jobs.graph.admission import (
    GRAPH_ADMISSION_DOMAIN_OWNER_ID,
    GraphAdmissionBuilder,
    GraphAdmissionCapabilityDomain,
    RegistryInitialTaskProposalResolver,
    graph_admission_declarations,
)
from ..jobs.graph.capabilities import (
    GRAPH_TASK_DOMAIN_OWNER_ID,
    GraphTaskCapabilityDomain,
    graph_task_capability_declarations,
)
from ..jobs.graph.models import (
    MAX_MODEL_REQUESTS_PER_ATTEMPT,
    TERMINAL_GRAPH_STATES,
    GraphAdmission,
    GraphInspection,
    GraphJob,
    GraphMutation,
    GraphState,
    GraphTask,
    TaskAttempt,
    TaskCheckpoint,
    TaskControl,
    TaskDependency,
    TaskResult,
    TaskState,
)
from ..jobs.owner import GraphBlockerProjection, JobOwner
from ..jobs.projections import GraphBoardProjection, GraphTimelinePage
from ..jobs.supervisor import JobSupervisor
from ..llm.models import (
    CanonicalMessage,
    ModelCallPolicy,
    ModelProfile,
    ModelSensitivity,
    ToolCall,
    ToolResultBlock,
)
from ..llm.profiles import reviewed_model_profile
from ..llm.protocols import ManagedModelProvider, ModelProvider
from ..llm.provider_definitions import (
    AuthenticationMode,
    admit_model_selection,
    provider_definition,
    tool_schema_incompatibility,
)
from ..llm.routing import (
    ModelRoute,
)
from ..llm.subscription_auth import CodexDevicePrompt, login_codex_subscription
from ..loop.driver import (
    AgentLoop,
    ContextBuilder,
    ToolRuntime,
)
from ..loop.models import (
    ConversationRun,
    LoopExit,
    LoopLimits,
    RunInput,
    RunOrigin,
    Transcript,
)
from ..loop.session import RunSession, RunSessionEvidence, RunSessionOptions
from ..loop.transcripts import ConversationPredecessor, RunSessionWriter
from ..memory import MemoryStore
from ..memory.capabilities import (
    MEMORY_DOMAIN_OWNER_ID,
    MemoryCapabilityDomain,
    memory_set_declarations,
)
from ..observation import (
    AgentObserver,
    PrivateDiagnosticSink,
    SupervisorStatus,
    observe_all,
)
from ..routines.capabilities import (
    ROUTINE_DOMAIN_OWNER_ID,
    ROUTINE_SAVE_TOOL_NAME,
    RoutineCapabilityDomain,
    routine_capability_declarations,
)
from ..routines.models import (
    MAX_ROUTINE_LIST_PAGE_SIZE,
    MisfirePolicy,
    ReportingMode,
    RequestedCapabilityGrant,
    ResourceRevisionObservation,
    ResourceRevisionPrecheck,
    RoutineControlAction,
    RoutineOccurrence,
    RoutineSchedule,
    RoutineState,
    ScheduledRoutine,
    ScheduledRoutineDraft,
    ScheduledRoutineInspection,
    ScheduledRoutineSummary,
)
from ..routines.owner import RoutineError, RoutineOwner, routine_approval_arguments
from ..routines.supervisor import RoutineSupervisor
from ..scope import (
    EffectiveSourceScope,
    resolve_effective_source_scope,
    resolve_machine_source_scope,
)
from ..security import (
    CredentialSession,
    KeychainSecretProvider,
    KeychainStore,
    SecretProvider,
    SecretReference,
    default_secret_provider,
)
from ..semantics import (
    SEMANTIC_DOMAIN_OWNER_ID,
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticAnnotationView,
    SemanticCapabilityDomain,
    SemanticKind,
    inspect_semantic_annotations,
    semantic_declarations,
)
from ..skills import Skill, SkillStore, SkillSummary
from ..skills.capabilities import (
    SKILL_DOMAIN_OWNER_ID,
    SkillCapabilityDomain,
    skill_declarations,
)
from ..storage.advisory import AdvisoryStorage
from ..storage.protocols import StateStore
from ..storage.sqlite import (
    SQLiteStateStore,
)
from ..storage.sqlite_records import (
    EffectReceipt,
    EffectResolution,
    EffectResolutionDecision,
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    SourceReadMode,
    validate_effect_receipt_id,
)
from ..work_board import (
    DEFAULT_DONE_WINDOW,
    MAX_DONE_WINDOW,
    WorkBoard,
    project_work_board,
)
from ..workspace import LocalWorkspace
from .agent_home import (
    _AGENT_NAME,
    _admit_agent_home,
    _candidate_agent_homes,
    _cleanup_failed_create,
    _delete_agent_home,
    _open_workspace_backend,
    _read_manifest,
    _reject_legacy_state_root,
    _require_unaliased_path,
    _resolve_state_root,
    _validate_agent_home_target,
    _validate_workspace_composition,
    _write_manifest,
    _WriterLock,
)
from .configuration import (
    _admit_host_model,
    _configuration_was_injected,
    _credential_account,
    _credential_account_prefix,
    _credential_reference_is_owned,
    _diagnostic_sink,
    _injected_concurrency_key,
    _keychain_references,
    _model_execution_contracts,
    _model_route,
    _owned_agent_credential_references,
    _owned_source_credential_references,
    _read_model_configuration,
    _read_model_configuration_document,
    _resolve_configuration,
    _resolve_credential_session,
    _stage_c_model_routes,
    _validate_custom_loop,
    _validate_model_route,
    _write_model_configuration,
)
from .errors import (
    AgentAlreadyExistsError,
    AgentHomeError,
    AgentIdentityMismatchError,
    AgentModelConfigurationError,
    AgentNameError,
    AgentNotConfiguredError,
    AgentNotFoundError,
    HostActiveError,
)
from .execution_governor import (
    AdmissionClosedError,
    CoordinatorDiagnostics,
    RunAdmissionCoordinator,
    WorkloadClass,
)
from .home_upgrade import AgentHomeStatus, inspect_agent_home, upgrade_agent_home
from .sources import (
    SourceEditConfirmationHandler,
    SourceEditPreview,
    SourceEditResult,
    _build_source_permissions_preview,
    _inspect_source_permissions,
    _resolve_source_selector,
    _source_edit_read_scope,
)

SourceBackends = Mapping[SourceOperation, Mapping[str, Any]]


def _backends_with(backends: SourceBackends, method: str) -> tuple[Any, ...]:
    """Each distinct source backend that implements one lifecycle method."""

    unique: dict[int, Any] = {}
    for by_adapter in backends.values():
        for backend in by_adapter.values():
            unique.setdefault(id(backend), backend)
    return tuple(
        backend
        for backend in unique.values()
        if callable(getattr(backend, method, None))
    )


async def _relational_upsert_readiness(
    data_view: CatalogDataView,
    backends: SourceBackends,
    agent_id: str,
    constraints: FrozenJsonObject,
) -> FrozenJsonObject:
    adapter_id = await data_view.source_adapter_id(
        agent_id, str(constraints["source_id"])
    )
    backend = (
        None
        if adapter_id is None
        else backends[SourceOperation.RELATIONAL_UPSERT].get(adapter_id)
    )
    if backend is None:
        raise CapabilityInputError(
            "resource_write_not_allowed",
            "Explicit current upsert permission is required.",
        )
    return cast(FrozenJsonObject, await backend.upsert_readiness(agent_id, constraints))


class _RoutineDraftValues(TypedDict):
    title: str
    authorized_instruction: str
    schedule: RoutineSchedule
    misfire_policy: MisfirePolicy
    reporting_mode: ReportingMode
    precheck: ResourceRevisionPrecheck | None
    allowed_source_ids: tuple[str, ...]
    allowed_connector_binding_ids: tuple[str, ...]
    allowed_resource_ids: tuple[str, ...]
    allowed_capability_ids: tuple[str, ...]
    sensitivity_ceiling: ModelSensitivity
    outcome_contract: OutcomeContract
    distribution_destination_id: str
    eligible_model_routes: tuple[str, ...]
    per_run_max_tokens: int
    per_run_max_cost_usd: Decimal | None
    cumulative_max_tokens: int
    cumulative_max_cost_usd: Decimal | None
    cumulative_max_attempts: int
    cumulative_max_occurrences: int
    maximum_consecutive_failures: int
    expires_at: datetime
    skill_names: tuple[str, ...]
    requested_capability_grants: tuple[RequestedCapabilityGrant, ...]
    run_immediately: bool


def _routine_draft_values(draft: ScheduledRoutineDraft) -> _RoutineDraftValues:
    return {
        "title": draft.title,
        "authorized_instruction": draft.authorized_instruction,
        "schedule": draft.schedule,
        "misfire_policy": draft.misfire_policy,
        "reporting_mode": draft.reporting_mode,
        "precheck": draft.precheck,
        "allowed_source_ids": draft.allowed_source_ids,
        "allowed_connector_binding_ids": draft.allowed_connector_binding_ids,
        "allowed_resource_ids": draft.allowed_resource_ids,
        "allowed_capability_ids": draft.allowed_capability_ids,
        "sensitivity_ceiling": draft.sensitivity_ceiling,
        "outcome_contract": draft.outcome_contract,
        "distribution_destination_id": draft.distribution_destination_id,
        "eligible_model_routes": draft.eligible_model_routes,
        "per_run_max_tokens": draft.per_run_max_tokens,
        "per_run_max_cost_usd": draft.per_run_max_cost_usd,
        "cumulative_max_tokens": draft.cumulative_max_tokens,
        "cumulative_max_cost_usd": draft.cumulative_max_cost_usd,
        "cumulative_max_attempts": draft.cumulative_max_attempts,
        "cumulative_max_occurrences": draft.cumulative_max_occurrences,
        "maximum_consecutive_failures": draft.maximum_consecutive_failures,
        "expires_at": draft.expires_at,
        "skill_names": draft.skill_names,
        "requested_capability_grants": draft.requested_capability_grants,
        "run_immediately": draft.run_immediately,
    }


_CONVERSATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_MAX_DISCOVERED_AGENTS = 100
_LIVE_JOB_STATES = frozenset(GraphState) - TERMINAL_GRAPH_STATES
# Scheduled and job runs may use every capability except these owners'
# (invariant 6): creating, changing or running routines and jobs stays with a
# person. Local computer files are excluded by scope: no session folder.
MACHINE_RUN_EXCLUDED_OWNERS = frozenset(
    {
        ROUTINE_DOMAIN_OWNER_ID,
        JOB_DOMAIN_OWNER_ID,
        GRAPH_ADMISSION_DOMAIN_OWNER_ID,
        DATA_PROFILE_DOMAIN_OWNER_ID,
    }
)


def machine_run_excluded_owners(origin: RunOrigin) -> frozenset[str]:
    """The exclusions for one run; graph task tools serve graph task runs only."""

    if origin is RunOrigin.JOB_TASK:
        return MACHINE_RUN_EXCLUDED_OWNERS
    return MACHINE_RUN_EXCLUDED_OWNERS | {GRAPH_TASK_DOMAIN_OWNER_ID}


_CREDENTIAL_CLEANUP_TIMEOUT_SECONDS = 1.0
_RUN_ADMISSION_DRAIN_SECONDS = 30.0
_T = TypeVar("_T")


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


async def _current_execution_contracts(
    store: StateStore,
    registry: CapabilityRegistry,
    model_contracts: Mapping[str, str],
    *,
    agent_id: str,
    source_ids: tuple[str, ...],
    resource_ids: tuple[str, ...],
    capability_ids: tuple[str, ...],
    connector_binding_ids: tuple[str, ...],
    model_route_ids: tuple[str, ...],
) -> ExecutionContractBindings:
    """Read exact current structural/admission references at the composition boundary."""

    identity = await store.load_identity()
    if identity is None or identity.id != agent_id:
        raise ValueError("execution contract agent identity changed")
    resources: dict[str, str] = {}
    for resource_id in resource_ids:
        resource = await store.load_resource(agent_id, resource_id)
        if (
            resource is None
            or resource.agent_id != agent_id
            or resource.source_id not in source_ids
        ):
            raise ValueError("execution contract resource identity changed")
        resources[resource_id] = resource.current_revision
    expected_origins = {
        view.capability_id: view.origin_revision_digest
        for name in registry.tool_names
        for view, _ in (registry.resolve_tool(name),)
        if view.origin_revision_digest is not None
        and view.capability_id in capability_ids
    }
    origins: dict[str, str] = {}
    for binding_id in connector_binding_ids:
        binding = await store.load_mcp_binding(agent_id, binding_id)
        if (
            binding is None
            or binding.agent_id != agent_id
            or binding.state is not MCPBindingState.ACTIVE
        ):
            raise ValueError("execution contract MCP binding is unavailable")
        for tool in binding.tools:
            if tool.capability_id not in capability_ids:
                continue
            origin = mcp_execution_origin_digest(binding, tool)
            if (
                expected_origins.get(tool.capability_id) != origin
                or tool.capability_id in origins
            ):
                raise ValueError("execution contract MCP origin changed")
            origins[tool.capability_id] = origin
    result = ExecutionContractBindings(
        capability_contracts={
            capability_id: registry.contract_digest(capability_id)
            for capability_id in capability_ids
        },
        tool_origins=origins,
        resource_revisions=resources,
        model_routes={
            route_id: model_contracts[route_id] for route_id in model_route_ids
        },
    )
    result.validate_coverage(
        capability_ids=capability_ids,
        resource_ids=resource_ids,
        route_ids=model_route_ids,
        mcp_capability_ids=expected_origins,
    )
    return result


def _validate_conversation_id(value: str) -> None:
    if not isinstance(value, str) or _CONVERSATION_ID.fullmatch(value) is None:
        raise ValueError("conversation_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


class _DeclarationSet(Protocol):
    @property
    def capabilities(self) -> tuple[Capability, ...]: ...

    @property
    def tool_views(self) -> tuple[ToolView, ...]: ...


def _domain_declarations(
    domain_owner_id: str, *sets: _DeclarationSet | None
) -> CapabilityDeclarations:
    """Bundle one domain's capabilities and views from its optional declaration sets."""
    present = [item for item in sets if item is not None]
    capabilities = tuple(
        capability for item in present for capability in item.capabilities
    )
    return CapabilityDeclarations(
        domain_owner_id=domain_owner_id,
        capabilities=capabilities,
        executor_ids=tuple(capability.executor_id for capability in capabilities),
        tool_views=tuple(view for item in present for view in item.tool_views),
    )


class EmbeddedAgent:
    """Identity, catalog, tools, transcript loop, and nothing else."""

    def __init__(
        self,
        *,
        identity: AgentIdentity,
        home: Path | None,
        workspace: LocalWorkspace | None,
        workspace_backend: LocalWorkspaceBackend | None,
        writer_lock: _WriterLock | None,
        store: StateStore,
        distribution_owner: DistributionOwner,
        loop: AgentLoop | None,
        transcripts: StateStore,
        capabilities: CapabilityRegistry,
        catalog_service: CatalogService,
        data_view: CatalogDataView,
        capability_runtime: CapabilityRuntime,
        approval_handler: ApprovalHandler | None,
        job_owner: JobOwner,
        routine_owner: RoutineOwner,
        routine_supervisor: RoutineSupervisor,
        job_supervisor: JobSupervisor,
        data_profile_admission: DataProfileAdmission,
        execution_contract_reader: ExecutionContractReader,
        data_profile_job_domain: DataProfileCapabilityDomain,
        semantic_domain: SemanticCapabilityDomain,
        memory_store: MemoryStore,
        skill_store: SkillStore,
        context_builder: AgentContextBuilder | None,
        artifact_store: AgentHomeArtifactStore,
        artifact_delivery: LocalArtifactDelivery | None,
        admission_coordinator: RunAdmissionCoordinator,
        host_capacity: HostCapacityLimits,
        source_types: SourceTypeRegistry,
        source_backends: SourceBackends,
        combined_query_engine: LocalFileQueryBackend | None,
        model_profile: ModelProfile | None,
        model_route: ModelRoute | None,
        owned_model_provider: ManagedModelProvider | None,
        limits: LoopLimits,
        model_call_policy: ModelCallPolicy,
        secret_provider: SecretProvider,
        keychain: CredentialSession,
        owns_credential_session: bool,
        model_validator: ModelProvider | None,
        mcp_client_factory: MCPClientFactory,
        mcp_connection_provider: MCPConnectionProvider | None,
        mcp_activated_bindings: tuple[MCPActivatedBinding, ...],
        stage_mcp_catalog: Callable[
            [tuple[MCPServerBinding, ...], tuple[MCPActivatedBinding, ...]],
            Awaitable[tuple[tuple[MCPActivatedBinding, ...], Callable[[], None]]],
        ],
        hosted: bool,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        close_state: bool = True,
    ) -> None:
        self.identity = identity
        self._home = home
        self._close_state = close_state
        self._workspace = workspace
        self._workspace_backend = workspace_backend
        self.model_profile = model_profile
        self.model_route = model_route
        self._owned_model_provider = owned_model_provider
        self._limits = limits
        self._model_call_policy = model_call_policy
        self._secret_provider = secret_provider
        self._keychain = keychain
        self._owns_credential_session = owns_credential_session
        self._model_validator = model_validator
        self._mcp_client_factory = mcp_client_factory
        self._mcp_connection_provider = mcp_connection_provider
        self._stage_mcp_catalog = stage_mcp_catalog
        self._hosted = hosted
        self._mcp_activated_bindings = {
            item.binding.binding_id: item for item in mcp_activated_bindings
        }
        self._retired_mcp_bindings: list[MCPActivatedBinding] = []
        self._writer_lock = writer_lock
        self._store = store
        self._distribution_owner = distribution_owner
        self._loop = loop
        self._transcripts = transcripts
        self._capabilities = capabilities
        self._catalog_service = catalog_service
        self._data_view = data_view
        self._capability_runtime = capability_runtime
        self._approval_handler = approval_handler
        self._job_owner = job_owner
        self._routine_owner = routine_owner
        self._routine_supervisor = routine_supervisor
        self._job_supervisor = job_supervisor
        self._data_profile_admission = data_profile_admission
        self._execution_contract_reader = execution_contract_reader
        self._data_profile_job_domain = data_profile_job_domain
        self._semantic_domain = semantic_domain
        self._memory_store = memory_store
        self._skill_store = skill_store
        self._context_builder = context_builder
        self._artifact_store = artifact_store
        self._artifact_delivery = artifact_delivery
        self._clock = clock
        self._id_factory = id_factory
        self._admission_coordinator = admission_coordinator
        self._host_capacity = host_capacity
        self._source_types = source_types
        self._source_backends = source_backends
        self._combined_query_engine = combined_query_engine
        self._model_management_lock = asyncio.Lock()
        self._effect_resolution_lock = asyncio.Lock()
        self._routine_management_lock = asyncio.Lock()
        self._artifact_publication_lock = (
            artifact_delivery.publication_lock
            if artifact_delivery is not None
            else asyncio.Lock()
        )
        self._semantic_management_lock = asyncio.Lock()
        self._credential_management_lock = asyncio.Lock()
        self._mcp_management_lock = asyncio.Lock()
        self._mcp_commit_lock = asyncio.Lock()
        self._source_management_lock = asyncio.Lock()
        self._source_commit_lock = asyncio.Lock()
        self._source_permission_lock = asyncio.Lock()
        self._source_permission_confirmation_key = secrets.token_bytes(32)
        self._source_permission_previews: dict[str, SourcePermissionsPreview] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._model_reopen_required = False

    @property
    def home(self) -> Path:
        if self._home is None:
            raise AgentHomeError(
                "externally stored agents have no local home; configuration belongs to the caller"
            )
        return self._home

    @property
    def session_source_id(self) -> str | None:
        """The live session folder source of this open agent, if it has one."""

        return self._data_view.session_source_id

    @property
    def workspace(self) -> LocalWorkspace:
        if self._workspace is None:
            raise AgentHomeError("hosted composition has no local file access")
        return self._workspace

    def _resolve_caller_principal(self, caller_principal_id: str | None) -> str:
        """Accept only host-supplied caller identity, never model content."""
        if caller_principal_id is None:
            return self.identity.id
        if (
            not isinstance(caller_principal_id, str)
            or not caller_principal_id
            or len(caller_principal_id) > 512
            or any(c in caller_principal_id for c in "\r\n\x00")
        ):
            raise ValueError("caller principal is invalid")
        return caller_principal_id

    @classmethod
    async def list(
        cls,
        *,
        root: str | Path | None = None,
    ) -> tuple[str, ...]:
        """Return a deterministic bounded view of valid local agent homes."""

        homes = await asyncio.to_thread(_candidate_agent_homes, root)
        names: list[str] = []
        for home in homes:
            try:
                manifest = await asyncio.to_thread(_read_manifest, home, home.name)
                identity = await SQLiteStateStore(home / "state.db").load_identity()
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            if identity != manifest:
                continue
            names.append(manifest.display_name)
            if len(names) == _MAX_DISCOVERED_AGENTS:
                break
        return tuple(names)

    @classmethod
    async def inspect_home(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
    ) -> AgentHomeStatus:
        """Read the persisted revision and recovery state without upgrading it."""

        if not isinstance(name, str) or _AGENT_NAME.fullmatch(name) is None:
            raise AgentNameError(
                "agent name must contain 1-64 ASCII letters, digits, '_' or '-'"
            )
        state_root = _resolve_state_root(root)
        _reject_legacy_state_root(state_root)
        agents_root = state_root / "agents"
        home = agents_root / name
        if not home.is_dir():
            raise AgentNotFoundError(f"agent does not exist: {name}")
        _require_unaliased_path(agents_root, "agents directory")
        home = _require_unaliased_path(home, "agent home")
        return await asyncio.to_thread(inspect_agent_home, home)

    @classmethod
    async def delete(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        keychain: KeychainStore | None = None,
        config: AgentConfig | None = None,
    ) -> None:
        """Delete one inactive agent home after removing its owned credentials."""

        _, cancelled = await _await_async_completion(
            lambda: cls._delete_admitted_home(
                name,
                root=root,
                keychain=keychain or KeychainSecretProvider(),
                source_types=SourceTypeRegistry(
                    () if config is None else config.source_types
                ),
            )
        )
        if cancelled:
            raise asyncio.CancelledError

    @classmethod
    async def _delete_admitted_home(
        cls,
        name: str,
        *,
        root: str | Path | None,
        keychain: KeychainStore,
        source_types: SourceTypeRegistry,
    ) -> None:
        del cls
        (home, writer_lock), _cancelled = await _await_sync_completion(
            lambda: _admit_agent_home(name, root, False)
        )
        store: StateStore | None = None
        try:
            manifest, _cancelled = await _await_sync_completion(
                lambda: _read_manifest(home, name)
            )
            store = SQLiteStateStore(home / "state.db")
            try:
                (
                    identity,
                    source_configurations,
                    mcp_reference_values,
                ) = await store.load_deletion_credential_inventory(manifest.id)
            except (OSError, sqlite3.Error, TypeError, ValueError) as error:
                raise AgentHomeError(
                    "agent credential inventory is invalid; "
                    "the agent home was preserved"
                ) from error
            if identity is None or identity != manifest:
                raise AgentIdentityMismatchError(
                    "agent.toml does not match state.db identity"
                )
            model_document, _cancelled = await _await_sync_completion(
                lambda: _read_model_configuration_document(home)
            )
            references = _owned_agent_credential_references(
                identity.id,
                model_document=model_document,
                source_configurations=source_configurations,
                source_types=source_types,
                mcp_reference_values=mcp_reference_values,
            )
            failures = 0
            for reference in references:
                try:
                    await keychain.delete(reference)
                except Exception:
                    failures += 1
            if failures:
                raise AgentHomeError(
                    "agent credentials could not all be deleted; "
                    "the agent home was preserved"
                )
            await store.close()
            store = None
            await _await_sync_completion(lambda: _delete_agent_home(home))
        finally:
            if store is not None:
                await store.close()
            writer_lock.release()

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        workspace: LocalWorkspace | None = None,
        hosted: bool = False,
        root: str | Path | None = None,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        tools: ToolRuntime | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        mcp_client_factory: MCPClientFactory | None = None,
        mcp_connection_provider: MCPConnectionProvider | None = None,
        keychain: KeychainStore | None = None,
        model_validator: ModelProvider | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        downloads_directory: Path | None = None,
        diagnostics_path: Path | None = None,
    ) -> Self:
        _validate_workspace_composition(workspace, hosted=hosted)
        if downloads_directory is not None and not isinstance(
            downloads_directory, Path
        ):
            raise TypeError("downloads_directory must be pathlib.Path or None")
        diagnostics = _diagnostic_sink(diagnostics_path)
        resolved_clock = clock or _utc_now
        resolved_ids = id_factory or _new_id
        credential_session, owns_credential_session = _resolve_credential_session(
            keychain
        )
        runtime_secrets = secret_provider or credential_session
        model, model_profile, model_route, limits, model_call_policy = (
            _resolve_configuration(
                config,
                model=model,
                model_profile=model_profile,
                limits=limits,
                secret_provider=runtime_secrets,
            )
        )
        _validate_custom_loop(model, model_profile, context_builder, tools)
        (home, writer_lock), cancelled = await _await_sync_completion(
            lambda: _admit_agent_home(name, root, True)
        )
        if cancelled:
            writer_lock.release()
            raise asyncio.CancelledError
        composition_cleanup = AsyncExitStack()
        store: StateStore | None = None
        workspace_backend: LocalWorkspaceBackend | None = None
        published = False
        try:
            workspace_backend = await _open_workspace_backend(
                workspace,
                hosted=hosted,
                home=home,
                clock=resolved_clock,
            )
            if any(
                path.exists() or path.is_symlink()
                for path in (home / "agent.toml", home / "state.db")
            ):
                raise AgentAlreadyExistsError(f"agent already exists: {name}")
            identity = AgentIdentity(
                id=resolved_ids("agent"),
                display_name=name,
                created_at=resolved_clock(),
            )
            store = await SQLiteStateStore.open(home / "state.db", clock=resolved_clock)
            await store.initialize_identity(identity)
            artifact_store = await AgentHomeArtifactStore.open(
                agent_id=identity.id,
                agent_home=home,
                registry=store,
                clock=resolved_clock,
                id_factory=resolved_ids,
            )
            artifact_delivery = (
                None
                if hosted
                else await LocalArtifactDelivery.open(
                    agent_id=identity.id,
                    agent_home=home,
                    artifacts=artifact_store,
                    sources=store,
                    exact_target_resolver=workspace_backend,
                    downloads_directory=downloads_directory,
                    clock=resolved_clock,
                    id_factory=resolved_ids,
                )
            )
            embedded = await cls._compose(
                identity=identity,
                home=home,
                workspace=workspace,
                workspace_backend=workspace_backend,
                hosted=hosted,
                writer_lock=writer_lock,
                store=store,
                model=model,
                model_profile=model_profile,
                model_route=model_route,
                context_builder=context_builder,
                tools=tools,
                limits=limits,
                analysis_limits=None if config is None else config.analysis_limits,
                host_capacity=None if config is None else config.host_capacity,
                postgresql_reads=None if config is None else config.postgresql_reads,
                source_types=() if config is None else config.source_types,
                admission_observations=(
                    False if config is None else config.admission_observations
                ),
                model_call_policy=model_call_policy,
                clock=resolved_clock,
                id_factory=resolved_ids,
                secret_provider=runtime_secrets,
                mcp_client_factory=mcp_client_factory,
                mcp_connection_provider=mcp_connection_provider,
                keychain=credential_session,
                owns_credential_session=owns_credential_session,
                model_validator=model_validator,
                observer=observer,
                diagnostics=diagnostics,
                approval_handler=approval_handler,
                artifact_store=artifact_store,
                artifact_delivery=artifact_delivery,
                admission_cleanup=composition_cleanup,
            )
            composition_cleanup.pop_all()
            _, cancelled = await _await_sync_completion(
                lambda: _write_manifest(home, identity)
            )
            published = True
            if cancelled:
                raise asyncio.CancelledError
            return embedded
        except BaseException:
            await composition_cleanup.aclose()
            try:
                if workspace_backend is not None:
                    await workspace_backend.close()
                if store is not None:
                    await store.close()
            finally:
                if not published:
                    await _await_sync_completion(lambda: _cleanup_failed_create(home))
                writer_lock.release()
            raise

    @classmethod
    async def open(
        cls,
        name: str,
        *,
        workspace: LocalWorkspace | None = None,
        hosted: bool = False,
        root: str | Path | None = None,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        context_builder: ContextBuilder | None = None,
        tools: ToolRuntime | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        mcp_client_factory: MCPClientFactory | None = None,
        mcp_connection_provider: MCPConnectionProvider | None = None,
        keychain: KeychainStore | None = None,
        model_validator: ModelProvider | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        downloads_directory: Path | None = None,
        diagnostics_path: Path | None = None,
    ) -> Self:
        _validate_workspace_composition(workspace, hosted=hosted)
        if downloads_directory is not None and not isinstance(
            downloads_directory, Path
        ):
            raise TypeError("downloads_directory must be pathlib.Path or None")
        diagnostics = _diagnostic_sink(diagnostics_path)
        resolved_clock = clock or _utc_now
        resolved_ids = id_factory or _new_id
        credential_session, owns_credential_session = _resolve_credential_session(
            keychain
        )
        runtime_secrets = secret_provider or credential_session
        limit_override = limits
        explicit_configuration = _configuration_was_injected(
            config,
            model=model,
            model_profile=model_profile,
            context_builder=context_builder,
            tools=tools,
        )
        if explicit_configuration:
            model, model_profile, model_route, limits, model_call_policy = (
                _resolve_configuration(
                    config,
                    model=model,
                    model_profile=model_profile,
                    limits=limits,
                    secret_provider=runtime_secrets,
                )
            )
        else:
            model = None
            model_profile = None
            model_route = None
            limits = limit_override or LoopLimits()
        _validate_custom_loop(model, model_profile, context_builder, tools)
        (home, writer_lock), cancelled = await _await_sync_completion(
            lambda: _admit_agent_home(name, root, False)
        )
        if cancelled:
            writer_lock.release()
            raise asyncio.CancelledError
        composition_cleanup = AsyncExitStack()
        store: StateStore | None = None
        workspace_backend: LocalWorkspaceBackend | None = None
        try:
            manifest, cancelled = await _await_sync_completion(
                lambda: _read_manifest(home, name)
            )
            if cancelled:
                raise asyncio.CancelledError
            _, cancelled = await _await_sync_completion(
                lambda: upgrade_agent_home(
                    home,
                    validate_home=lambda source, candidate, affected: (
                        _validate_agent_home_target(
                            manifest,
                            source,
                            candidate,
                            affected,
                        )
                    ),
                )
            )
            if cancelled:
                raise asyncio.CancelledError
            workspace_backend = await _open_workspace_backend(
                workspace,
                hosted=hosted,
                home=home,
                clock=resolved_clock,
            )
            store = await SQLiteStateStore.open(
                home / "state.db",
                clock=resolved_clock,
                current_home_validated=True,
            )
            identity = await store.load_identity()
            if identity is None or identity != manifest:
                raise AgentIdentityMismatchError(
                    "agent.toml does not match state.db identity"
                )
            await store.list_sources(identity.id)
            recovery_task = asyncio.create_task(
                _recover_analysis_evidence(store, identity.id)
            )
            try:
                await asyncio.shield(recovery_task)
            except asyncio.CancelledError:
                await recovery_task
                raise
            await _recover_agent_state(store, identity.id, resolved_clock)
            if not explicit_configuration:
                persisted, cancelled = await _await_sync_completion(
                    lambda: _read_model_configuration(home, identity.id)
                )
                if cancelled:
                    raise asyncio.CancelledError
                model, model_profile, model_route, limits, model_call_policy = (
                    _resolve_configuration(
                        persisted,
                        model=None,
                        model_profile=None,
                        limits=limit_override,
                        secret_provider=runtime_secrets,
                    )
                )
            artifact_store = await AgentHomeArtifactStore.open(
                agent_id=identity.id,
                agent_home=home,
                registry=store,
                clock=resolved_clock,
                id_factory=resolved_ids,
            )
            artifact_delivery = (
                None
                if hosted
                else await LocalArtifactDelivery.open(
                    agent_id=identity.id,
                    agent_home=home,
                    artifacts=artifact_store,
                    sources=store,
                    exact_target_resolver=workspace_backend,
                    downloads_directory=downloads_directory,
                    clock=resolved_clock,
                    id_factory=resolved_ids,
                )
            )
            embedded = await cls._compose(
                identity=identity,
                home=home,
                workspace=workspace,
                workspace_backend=workspace_backend,
                hosted=hosted,
                writer_lock=writer_lock,
                store=store,
                model=model,
                model_profile=model_profile,
                model_route=model_route,
                context_builder=context_builder,
                tools=tools,
                limits=limits,
                analysis_limits=None if config is None else config.analysis_limits,
                host_capacity=None if config is None else config.host_capacity,
                postgresql_reads=None if config is None else config.postgresql_reads,
                source_types=() if config is None else config.source_types,
                admission_observations=(
                    False if config is None else config.admission_observations
                ),
                model_call_policy=model_call_policy,
                clock=resolved_clock,
                id_factory=resolved_ids,
                secret_provider=runtime_secrets,
                mcp_client_factory=mcp_client_factory,
                mcp_connection_provider=mcp_connection_provider,
                keychain=credential_session,
                owns_credential_session=owns_credential_session,
                model_validator=model_validator,
                observer=observer,
                diagnostics=diagnostics,
                approval_handler=approval_handler,
                artifact_store=artifact_store,
                artifact_delivery=artifact_delivery,
                admission_cleanup=composition_cleanup,
            )
            composition_cleanup.pop_all()
            return embedded
        except BaseException:
            await composition_cleanup.aclose()
            if workspace_backend is not None:
                await workspace_backend.close()
            if store is not None:
                await store.close()
            writer_lock.release()
            raise

    @classmethod
    async def from_storage(
        cls,
        *,
        agent_id: str,
        state: StateStore,
        advisory_storage: AdvisoryStorage,
        artifact_storage: ArtifactByteStorage,
        config: AgentConfig | None = None,
        model: ModelProvider | None = None,
        model_profile: ModelProfile | None = None,
        limits: LoopLimits | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        mcp_client_factory: MCPClientFactory | None = None,
        mcp_connection_provider: MCPConnectionProvider | None = None,
        keychain: KeychainStore | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        diagnostics_path: Path | None = None,
    ) -> Self:
        """Compose the existing runtime over caller-admitted, borrowed storage.

        The caller must validate/upgrade all storage, initialize identity and hold
        exclusive execution ownership before this call. Recovery runs here, once
        before supervisors start. Keep supplied resources and ownership alive
        throughout admission and, on success, until close succeeds. No local
        home, configuration persistence, provisioning or lease is created here.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")
        diagnostics = _diagnostic_sink(diagnostics_path)
        resolved_clock = clock or _utc_now
        resolved_ids = id_factory or _new_id

        async def compose() -> Self:
            credential_session, owns_credentials = _resolve_credential_session(keychain)
            runtime_secrets = secret_provider or credential_session
            async with AsyncExitStack() as cleanup:
                if owns_credentials:
                    cleanup.push_async_callback(credential_session.close)
                identity = await state.load_identity()
                if identity is None or identity.id != agent_id:
                    raise AgentIdentityMismatchError(
                        "storage does not match the requested agent identity"
                    )
                resolved_model, profile, route, resolved_limits, policy = (
                    _resolve_configuration(
                        config,
                        model=model,
                        model_profile=model_profile,
                        limits=limits,
                        secret_provider=runtime_secrets,
                    )
                )
                if route is not None:
                    assert isinstance(resolved_model, ManagedModelProvider)
                    cleanup.push_async_callback(resolved_model.close)
                await state.list_sources(identity.id)
                await _recover_agent_state(state, identity.id, resolved_clock)
                artifact_store = await AgentHomeArtifactStore.open(
                    agent_id=identity.id,
                    registry=state,
                    byte_storage=artifact_storage,
                    clock=resolved_clock,
                    id_factory=resolved_ids,
                )
                cleanup.push_async_callback(artifact_store.close)
                return await cls._compose(
                    identity=identity,
                    home=None,
                    workspace=None,
                    workspace_backend=None,
                    hosted=True,
                    writer_lock=None,
                    store=state,
                    close_state=False,
                    model=resolved_model,
                    model_profile=profile,
                    model_route=route,
                    context_builder=None,
                    tools=None,
                    limits=resolved_limits,
                    host_capacity=None if config is None else config.host_capacity,
                    postgresql_reads=(
                        None if config is None else config.postgresql_reads
                    ),
                    source_types=() if config is None else config.source_types,
                    admission_observations=(
                        False if config is None else config.admission_observations
                    ),
                    model_call_policy=policy,
                    clock=resolved_clock,
                    id_factory=resolved_ids,
                    secret_provider=runtime_secrets,
                    mcp_client_factory=mcp_client_factory,
                    mcp_connection_provider=mcp_connection_provider,
                    keychain=credential_session,
                    owns_credential_session=owns_credentials,
                    model_validator=None,
                    observer=observer,
                    diagnostics=diagnostics,
                    approval_handler=approval_handler,
                    artifact_store=artifact_store,
                    artifact_delivery=None,
                    advisory_storage=advisory_storage,
                    admission_cleanup=cleanup,
                )

        embedded: Self
        embedded, cancelled = await _await_async_completion(compose)
        if cancelled:
            await embedded.close()
            raise asyncio.CancelledError
        return embedded

    @classmethod
    async def _compose(
        cls,
        *,
        identity: AgentIdentity,
        home: Path | None,
        workspace: LocalWorkspace | None,
        workspace_backend: LocalWorkspaceBackend | None,
        hosted: bool,
        writer_lock: _WriterLock | None,
        store: StateStore,
        model: ModelProvider | None,
        model_profile: ModelProfile | None,
        model_route: ModelRoute | None,
        context_builder: ContextBuilder | None,
        tools: ToolRuntime | None,
        limits: LoopLimits,
        analysis_limits: AnalysisLimits | None = None,
        host_capacity: HostCapacityLimits | None = None,
        postgresql_reads: PostgreSQLReadLimits | None = None,
        source_types: tuple[SourceType, ...] = (),
        admission_observations: bool = False,
        model_call_policy: ModelCallPolicy,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        secret_provider: SecretProvider,
        mcp_client_factory: MCPClientFactory | None,
        mcp_connection_provider: MCPConnectionProvider | None,
        keychain: CredentialSession,
        owns_credential_session: bool,
        model_validator: ModelProvider | None,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
        artifact_store: AgentHomeArtifactStore,
        diagnostics: PrivateDiagnosticSink | None = None,
        artifact_delivery: LocalArtifactDelivery | None,
        advisory_storage: AdvisoryStorage | None = None,
        close_state: bool = True,
        admission_cleanup: AsyncExitStack | None = None,
    ) -> Self:
        owned_model_provider: ManagedModelProvider | None = None
        if model_route is not None:
            if model is None or not isinstance(model, ManagedModelProvider):
                raise AgentNotConfiguredError(
                    "configured model route did not produce a managed provider"
                )
            owned_model_provider = model
        # Hosted hosts have no local file authority: no folder type, no session source.
        registry = SourceTypeRegistry(source_types, local_files=not hosted)
        session_source = (
            None
            if workspace is None or workspace_backend is None
            else SessionFolderSource.for_workspace(identity.id, workspace)
        )
        catalog_service = CatalogService(store, store)
        data_view = CatalogDataView(
            store, catalog_service, store, registry, session_source=session_source
        )
        catalog = catalog_declarations(identity.id, data_view)
        host_capacity = host_capacity or HostCapacityLimits()
        backend_context = SourceBackendContext(
            store=store,
            catalog=data_view,
            secret_provider=secret_provider or keychain,
            clock=clock,
            host_capacity=host_capacity,
            postgresql_reads=postgresql_reads,
            deadline_reader=lambda: admission_coordinator.current_deadline(),
            observer=observer if admission_observations else None,
            home=home,
            session_workspace=workspace_backend,
            session_source_id=(
                None if session_source is None else session_source.source_id
            ),
        )
        source_backends: dict[SourceOperation, dict[str, Any]] = {
            operation: {} for operation in SourceOperation
        }
        for source_type in registry.types():
            for operation, backend in source_type.backends(backend_context).items():
                source_backends[operation][source_type.adapter_id] = backend
        if admission_cleanup is not None:
            for backend in _backends_with(source_backends, "close"):
                admission_cleanup.push_async_callback(backend.close)
        from ..adapters.analytical_workspace.runtime import runtime_status

        scientific_admitted = not hosted and runtime_status()["available"] is True
        admitted_analysis_limits = analysis_limits or AnalysisLimits()
        export_arrow_bounds = (
            (
                min(100_000, admitted_analysis_limits.table_rows),
                min(256, admitted_analysis_limits.table_columns),
                min(
                    admitted_analysis_limits.file_bytes,
                    admitted_analysis_limits.input_bytes,
                    admitted_analysis_limits.decoded_bytes,
                    admitted_analysis_limits.scratch_bytes,
                ),
            )
            if scientific_admitted
            else None
        )
        relational_export = (
            exact_export_declarations(
                identity.id,
                source_backends[SourceOperation.TABULAR_EXPORT],
                clock,
                arrow_bounds=export_arrow_bounds,
            )
            if artifact_store.available
            else None
        )
        # Combined queries join their inputs in the same private DuckDB worker
        # as file queries; it needs local scratch, so homeless hosts omit them.
        combined_engine = (
            None
            if home is None
            else LocalFileQueryBackend(
                scratch_parent=home / "combined-query-scratch",
                limits=LocalFileQueryLimits(
                    max_result_rows=DATA_QUERY_MAX_ROWS,
                    max_result_bytes=DATA_QUERY_MAX_BYTES,
                ),
            )
        )
        if combined_engine is not None and admission_cleanup is not None:
            admission_cleanup.push_async_callback(combined_engine.close)
        relational_query = data_query_declarations(
            identity.id,
            source_backends[SourceOperation.SQL_READ],
            combined_engine=combined_engine,
            tabular_backends=source_backends[SourceOperation.TABULAR_READ],
            path_backends=source_backends[SourceOperation.PATH_QUERY],
            file_outputs=(
                exact_export_outputs(allow_arrow=export_arrow_bounds is not None)
                if relational_export is not None
                else ()
            ),
        )
        relational_update_preview = relational_update_preview_declarations(
            identity.id,
            source_backends[SourceOperation.RELATIONAL_UPDATE],
            data_view.source_adapter_id,
        )
        relational_update = relational_update_declarations(
            identity.id,
            source_backends[SourceOperation.RELATIONAL_UPDATE],
            data_view.source_adapter_id,
        )
        relational_upsert = relational_upsert_declarations(
            identity.id,
            source_backends[SourceOperation.RELATIONAL_UPSERT],
            data_view.source_adapter_id,
        )
        routine_precheck = resource_revision_observation_declarations(
            agent_id=identity.id,
            catalog=data_view,
            clock=clock,
        )
        if hosted != (workspace is None and workspace_backend is None):
            raise AgentHomeError("workspace composition is inconsistent")
        if hosted != (artifact_delivery is None):
            raise AgentHomeError("artifact delivery composition is inconsistent")
        source_files = (
            source_file_declarations(
                identity.id,
                source_backends[SourceOperation.FILE_SEARCH],
                source_backends[SourceOperation.TEXT_READ],
            )
            if source_backends[SourceOperation.FILE_SEARCH]
            else None
        )
        memory_lock = asyncio.Lock()
        skill_lock = asyncio.Lock()
        available_sources = tuple(
            registration
            for registration in await store.list_sources(identity.id)
            if registration.active and registry.get(registration.adapter_id)
        )
        for registration in available_sources:
            for scope in await store.list_relational_write_scopes(
                identity.id, registration.id
            ):
                if any(
                    not registry.supports(
                        registration.adapter_id, RELATIONAL_WRITE_OPERATIONS[name]
                    )
                    for name in scope.allowed_operations
                ):
                    raise AgentHomeError(
                        "stored relational write scope is foreign to its source type"
                    )
        source_capacities = {
            f"source:{registration.id}": registry.read_capacity(
                registration.adapter_id, host_capacity
            )
            for registration in available_sources
        }
        if session_source is not None:
            source_capacities[f"source:{session_source.source_id}"] = (
                registry.read_capacity(session_source.adapter_id, host_capacity)
            )
        admission_coordinator = RunAdmissionCoordinator(
            observer=observer if admission_observations else None,
            execution_capacity=host_capacity.execution_capacity,
            foreground_execution_reserve=host_capacity.foreground_execution_reserve,
            provider_capacity=host_capacity.provider_capacity,
            foreground_provider_reserve=host_capacity.foreground_provider_reserve,
            source_resource_capacities=source_capacities,
            sqlite_pressure_capacity=host_capacity.sqlite_pressure_capacity,
            analysis_worker_capacity=(
                1 if analysis_limits is None else analysis_limits.worker_count
            ),
        )
        declared_model = model
        model = _admit_host_model(
            model,
            admission_coordinator,
            identity=identity.id,
            home=home,
            profile=model_profile,
            owned=owned_model_provider is not None,
        )
        if owned_model_provider is not None:
            owned_model_provider = cast(ManagedModelProvider, model)
        memory_store = MemoryStore(home, memory_lock, storage=advisory_storage)
        skill_store = SkillStore(home, skill_lock, storage=advisory_storage)
        if advisory_storage is not None:
            await memory_store.read_context()
            await skill_store.list_skills()
        memory = memory_set_declarations(memory_store)
        skills = skill_declarations(skill_store)
        semantics = semantic_declarations(identity.id, store)
        artifacts = (
            artifact_declarations(
                artifact_delivery,
                artifact_store,
                workspace=workspace_backend,
                clock=clock,
            )
            if artifact_store.available
            else None
        )
        distribution_owner = DistributionOwner(agent_id=identity.id, store=store)
        job_owner = JobOwner(
            agent_id=identity.id,
            store=store,
            clock=clock,
            id_factory=id_factory,
        )
        job_lifecycle = job_capability_declarations(job_owner)
        graph_task_lifecycle = graph_task_capability_declarations(
            job_owner,
            store,
            clock=clock,
            id_factory=id_factory,
        )
        data_profile_jobs, data_profile_admission = data_profile_declarations(
            agent_id=identity.id,
            catalog=data_view,
            owner=job_owner,
            backends=source_backends[SourceOperation.PROFILE],
            source_types=registry,
            artifacts=artifact_store,
            distribution=distribution_owner,
            clock=clock,
            id_factory=id_factory,
        )
        data_declarations = _domain_declarations(
            DATA_DOMAIN_OWNER_ID,
            catalog,
            relational_query,
            relational_export,
            relational_update_preview,
            relational_update,
            relational_upsert,
            routine_precheck,
            source_files,
        )
        memory_declarations = _domain_declarations(MEMORY_DOMAIN_OWNER_ID, memory)
        skill_declaration_bundle = _domain_declarations(SKILL_DOMAIN_OWNER_ID, skills)
        semantic_declaration_bundle = _domain_declarations(
            SEMANTIC_DOMAIN_OWNER_ID, semantics
        )
        artifact_declaration_bundle = (
            None
            if artifacts is None
            else _domain_declarations(ARTIFACT_DOMAIN_OWNER_ID, artifacts)
        )
        job_declaration_bundle = _domain_declarations(
            JOB_DOMAIN_OWNER_ID, job_lifecycle
        )
        graph_task_declaration_bundle = _domain_declarations(
            GRAPH_TASK_DOMAIN_OWNER_ID, graph_task_lifecycle
        )
        data_profile_declaration_bundle = _domain_declarations(
            DATA_PROFILE_DOMAIN_OWNER_ID, data_profile_jobs
        )
        data_domain = DataCapabilityDomain(
            data_declarations,
            data_view,
            registry,
            relational_export_available=relational_export is not None,
            combined_query_available=combined_engine is not None,
            transcript_loader=store.load,
            upsert_readiness=lambda agent_id, constraints: (
                _relational_upsert_readiness(
                    data_view, source_backends, agent_id, constraints
                )
            ),
            source_files_available=source_files is not None,
            path_query_available=any(
                capability.id == PATH_QUERY_CAPABILITY_ID
                for capability in relational_query.capabilities
            ),
        )
        skill_domain = SkillCapabilityDomain(skill_declaration_bundle)
        semantic_domain = SemanticCapabilityDomain(
            semantic_declaration_bundle,
            data_view,
            store,
        )
        memory_domain = MemoryCapabilityDomain(
            memory_declarations,
            validate_schema_independent=semantic_domain.validate_schema_independent_memory,
        )
        artifact_domain = (
            None
            if artifact_declaration_bundle is None
            else ArtifactCapabilityDomain(
                artifact_declaration_bundle,
                data_view,
                store,
                artifact_store,
                artifact_delivery,
                workspace_backend,
                clock=clock,
            )
        )
        job_domain = JobCapabilityDomain(job_declaration_bundle, job_owner)
        graph_task_domain = GraphTaskCapabilityDomain(
            graph_task_declaration_bundle,
            job_owner,
        )
        data_profile_job_domain = DataProfileCapabilityDomain(
            data_profile_declaration_bundle,
            admission=data_profile_admission,
        )
        if model is not None and context_builder is None:
            assert model_profile is not None
            if not model_profile.supports_tools:
                raise AgentNotConfiguredError(
                    "the data agent requires a tool-capable model profile"
                )
        mcp_model_ids = (
            tuple(candidate.provider_id for candidate in model_route.candidates)
            if model_route is not None
            else (() if model_profile is None else (model_profile.id,))
        )
        resolved_mcp_client_factory: MCPClientFactory
        if mcp_client_factory is None:
            resolved_mcp_client_factory = SDKMCPClientFactory()
        else:
            resolved_mcp_client_factory = mcp_client_factory
        try:
            mcp_domain, mcp_activated_bindings, mcp_executors = (
                await activate_mcp_domain(
                    agent_id=identity.id,
                    store=store,
                    client_factory=resolved_mcp_client_factory,
                    secrets=secret_provider,
                    connection_provider=mcp_connection_provider,
                    clock=clock,
                    model_ids=mcp_model_ids,
                )
            )
        except MCPAdmissionError as error:
            if model_route is None or error.code != "mcp_schema_unsupported":
                raise
            raise AgentModelConfigurationError(
                "The saved model route is incompatible with attached MCP tools. "
                "Choose a compatible model or update MCP access."
            ) from error
        if admission_cleanup is not None:
            for activated in mcp_activated_bindings:
                admission_cleanup.push_async_callback(activated.executor.close)
        analysis_domain = (
            None
            if hosted
            else AnalysisCapabilityDomain(
                admission_coordinator,
                artifact_domain,
                artifact_store,
                observer=observer,
                **({"limits": analysis_limits} if analysis_limits is not None else {}),
            )
        )
        base_domains = (
            *((analysis_domain,) if analysis_domain is not None else ()),
            data_domain,
            memory_domain,
            skill_domain,
            semantic_domain,
            job_domain,
            graph_task_domain,
            data_profile_job_domain,
            *((artifact_domain,) if artifact_domain is not None else ()),
            *((mcp_domain,) if mcp_domain is not None else ()),
        )
        base_executors = (
            *((analysis_domain,) if analysis_domain is not None else ()),
            *catalog.executors,
            *relational_query.executors,
            *(relational_export.executors if relational_export is not None else ()),
            *relational_update_preview.executors,
            *relational_update.executors,
            *relational_upsert.executors,
            *routine_precheck.executors,
            *(source_files.executors if source_files is not None else ()),
            *memory.executors,
            *skills.executors,
            *semantics.executors,
            *job_lifecycle.executors,
            *graph_task_lifecycle.executors,
            *data_profile_jobs.executors,
            *(artifacts.executors if artifacts is not None else ()),
            *mcp_executors,
        )
        distribution_lifecycle = distribution_capability_declarations(
            distribution_owner
        )
        distribution_declaration_bundle = _domain_declarations(
            DISTRIBUTION_DOMAIN_OWNER_ID, distribution_lifecycle
        )
        distribution_domain = DistributionCapabilityDomain(
            distribution_declaration_bundle,
            distribution_owner,
        )
        # The host's retry wrapper is not caller configuration; frozen contracts
        # keep describing the declared provider and profile.
        model_contracts = _model_execution_contracts(
            declared_model, model_profile, model_route, model_call_policy
        )

        async def read_execution_contracts(
            *,
            agent_id: str,
            source_ids: tuple[str, ...],
            resource_ids: tuple[str, ...],
            capability_ids: tuple[str, ...],
            connector_binding_ids: tuple[str, ...],
            model_route_ids: tuple[str, ...],
        ) -> ExecutionContractBindings:
            return await _current_execution_contracts(
                store,
                capabilities,
                model_contracts,
                agent_id=agent_id,
                source_ids=source_ids,
                resource_ids=resource_ids,
                capability_ids=capability_ids,
                connector_binding_ids=connector_binding_ids,
                model_route_ids=model_route_ids,
            )

        graph_routes = _stage_c_model_routes(declared_model, model_route)
        routine_owner = RoutineOwner(
            execution_contract_reader=read_execution_contracts,
            agent_id=identity.id,
            store=store,
            catalog=data_view,
            distribution=distribution_owner,
            skills=skill_store,
            eligible_model_routes=graph_routes,
            maximum_per_run_tokens=limits.max_total_tokens,
            maximum_per_run_cost_usd=limits.max_estimated_cost_usd,
            clock=clock,
        )
        routine_lifecycle = routine_capability_declarations(routine_owner)
        routine_declaration_bundle = _domain_declarations(
            ROUTINE_DOMAIN_OWNER_ID, routine_lifecycle
        )
        routine_domain = RoutineCapabilityDomain(
            routine_declaration_bundle,
            routine_owner,
        )
        domains = (*base_domains, distribution_domain, routine_domain)
        registered_executors = (
            *base_executors,
            *distribution_lifecycle.executors,
            *routine_lifecycle.executors,
        )
        graph_builder: GraphAdmissionBuilder | None = None
        graph_resolver: RegistryInitialTaskProposalResolver | None = None
        if graph_routes:
            graph_prerequisite_registry = CapabilityRegistry(
                declarations=tuple(domain.declarations for domain in domains),
                executors=registered_executors,
            )
            graph_resolver = RegistryInitialTaskProposalResolver(
                agent_id=identity.id,
                registry=graph_prerequisite_registry,
                contract_reader=read_execution_contracts,
                model_route_id=graph_routes[0],
                max_steps=min(limits.max_steps, MAX_MODEL_REQUESTS_PER_ATTEMPT),
                per_run_max_tokens=limits.max_total_tokens,
            )
            graph_builder = GraphAdmissionBuilder(
                agent_id=identity.id,
                registry=graph_prerequisite_registry,
                distribution=distribution_owner,
                clock=clock,
                id_factory=id_factory,
            )
            graph_admission_bundle, graph_admission_executors = (
                graph_admission_declarations(
                    owner=job_owner,
                    resolver=graph_resolver,
                    builder=graph_builder,
                )
            )
            domains = (
                *domains,
                GraphAdmissionCapabilityDomain(graph_admission_bundle),
            )
            registered_executors = (
                *registered_executors,
                *graph_admission_executors,
            )
        if hosted:
            declared_capability_ids = {
                capability.id
                for domain in domains
                for capability in domain.declarations.capabilities
            }
            declared_view_ids = {
                view.capability_id
                for domain in domains
                for view in domain.declarations.tool_views
            }
            registered_executor_ids = {
                executor.executor_id for executor in registered_executors
            }
            if (
                declared_capability_ids & SOURCE_FILE_CAPABILITY_IDS
                or declared_view_ids & SOURCE_FILE_CAPABILITY_IDS
                or PATH_QUERY_EXECUTOR_ID in registered_executor_ids
                or session_source is not None
                or declared_capability_ids & LOCAL_ARTIFACT_EDIT_CAPABILITY_IDS
                or declared_view_ids & LOCAL_ARTIFACT_EDIT_CAPABILITY_IDS
                or registered_executor_ids & LOCAL_ARTIFACT_EDIT_EXECUTOR_IDS
                or workspace_backend is not None
                or artifact_delivery is not None
            ):
                raise AgentHomeError("hosted composition contains local file authority")
        capabilities = CapabilityRegistry(
            declarations=tuple(domain.declarations for domain in domains),
            executors=registered_executors,
        )
        if artifact_domain is not None:
            artifact_domain.bind_capability_registry(capabilities)
        data_domain.bind_capability_registry(capabilities)
        routine_owner.bind_capability_registry(capabilities)
        graph_task_lifecycle.bind_capability_registry(capabilities)

        async def resolve_run_sources(run: RunInput):
            return await resolve_effective_source_scope(run, data_view)

        # The one loop and runtime serve foreground, routine and graph-task runs,
        # so the host diagnostic sink sees every origin beside the caller observer.
        run_observer = observe_all(observer, diagnostics)

        capability_runtime = CapabilityRuntime(
            capabilities,
            domains,
            effect_receipts=store,
            execution_contract_reader=read_execution_contracts,
            approval_handler=approval_handler,
            admission_coordinator=admission_coordinator,
            effect_coordinator=admission_coordinator.effect_coordinator,
            owner_management_locks={
                MEMORY_DOMAIN_OWNER_ID: memory_lock,
                SKILL_DOMAIN_OWNER_ID: skill_lock,
            },
            source_scope_resolver=resolve_run_sources,
            observer=run_observer,
            clock=clock,
            artifacts=artifact_store,
            limits=limits,
        )
        routine_owner.bind_grant_preparer(capability_runtime.prepare_automation_grant)
        if graph_builder is not None:
            graph_builder.bind_grant_preparer(
                capability_runtime.prepare_automation_grant
            )
        native_domains = tuple(
            domain
            for domain in domains
            if domain.domain_owner_id != MCP_DOMAIN_OWNER_ID
        )
        initial_mcp_executor_ids = {item.executor_id for item in mcp_executors}
        native_executors = tuple(
            executor
            for executor in registered_executors
            if executor.executor_id not in initial_mcp_executor_ids
        )

        async def stage_mcp_catalog(
            bindings: tuple[MCPServerBinding, ...],
            existing: tuple[MCPActivatedBinding, ...],
        ) -> tuple[tuple[MCPActivatedBinding, ...], Callable[[], None]]:
            """Stage without I/O, then publish synchronously under system admission."""
            next_domain, next_bindings, next_executors = await activate_mcp_domain(
                agent_id=identity.id,
                store=store,
                client_factory=resolved_mcp_client_factory,
                secrets=secret_provider,
                connection_provider=mcp_connection_provider,
                clock=clock,
                bindings=bindings,
                existing=existing,
                model_ids=mcp_model_ids,
            )
            next_domains = (
                *native_domains,
                *((next_domain,) if next_domain is not None else ()),
            )
            next_registry = CapabilityRegistry(
                declarations=tuple(domain.declarations for domain in next_domains),
                executors=(*native_executors, *next_executors),
            )
            capability_runtime._catalog_owners(next_registry, next_domains)

            def publish() -> None:
                nonlocal capabilities
                # No await: every reader switches before another task can run.
                capability_runtime._replace_catalog(next_registry, next_domains)
                data_domain.bind_capability_registry(next_registry)
                if artifact_domain is not None:
                    artifact_domain.bind_capability_registry(next_registry)
                routine_owner.bind_capability_registry(next_registry)
                graph_task_lifecycle.bind_capability_registry(next_registry)
                if graph_resolver is not None:
                    graph_resolver.bind_capability_registry(next_registry)
                if graph_builder is not None:
                    graph_builder.bind_capability_registry(next_registry)
                capabilities = next_registry
                embedded._capabilities = next_registry
                embedded._mcp_activated_bindings = {
                    item.binding.binding_id: item for item in next_bindings
                }

            return next_bindings, publish

        resolved_context = context_builder
        resolved_tools = tools
        if model is not None and resolved_context is None:
            assert model_profile is not None
            resolved_context = AgentContextBuilder(
                data_view,
                profile=model_profile,
                user_timezone=(lambda: None) if hosted else None,
                routine_authoring_facts=routine_owner.authoring_facts,
                effect_receipts=store,
                memory=memory_store,
                skills=skill_store,
                semantics=store,
                artifact_destinations=(
                    artifact_delivery if artifacts is not None else None
                ),
                workspace_id=(
                    None
                    if workspace_backend is None
                    else workspace_backend.workspace_id
                ),
                workspace_sensitivity=(
                    None if workspace_backend is None else workspace_backend.sensitivity
                ),
                local_file_context=(
                    None
                    if workspace_backend is None or session_source is None
                    else FrozenJsonObject.from_mapping(
                        {
                            **workspace_backend.model_context().to_dict(),
                            "source_id": session_source.source_id,
                        }
                    )
                ),
                max_context_evidence_bytes=limits.max_context_evidence_bytes,
            )
            resolved_tools = capability_runtime
        transcripts = store
        loop = (
            None
            if model is None
            else AgentLoop(
                model=model,
                context_builder=_require_value(resolved_context),
                tools=_require_value(resolved_tools),
                transcripts=transcripts,
                limits=limits,
                model_call_policy=model_call_policy,
                clock=clock,
                observer=run_observer,
                stream_model_calls=(
                    model_profile.supports_streaming
                    if model_profile is not None
                    else False
                ),
            )
        )

        async def machine_run_scope(
            origin: RunOrigin,
            source_ids: tuple[str, ...],
            model_route_ids: tuple[str, ...],
        ) -> MachineRunScope:
            """Fix what a routine or job run may use as it starts (invariant 6)."""

            sources = await resolve_machine_source_scope(
                identity.id, source_ids, data_view
            )
            # The scope lists sources and resources together; a source with
            # nothing readable contributes nothing.
            source_tuple = (
                tuple(sorted(sources.source_ids)) if sources.resource_ids else ()
            )
            resource_tuple = tuple(sorted(sources.resource_ids))
            capability_ids = machine_run_capability_ids(
                capabilities, machine_run_excluded_owners(origin)
            )
            connector_ids = machine_run_connector_binding_ids(capabilities)
            return MachineRunScope(
                source_ids=source_tuple,
                resource_ids=resource_tuple,
                capability_ids=capability_ids,
                connector_binding_ids=connector_ids,
                contract_bindings=await read_execution_contracts(
                    agent_id=identity.id,
                    source_ids=source_tuple,
                    resource_ids=resource_tuple,
                    capability_ids=capability_ids,
                    connector_binding_ids=connector_ids,
                    model_route_ids=model_route_ids,
                ),
            )

        job_supervisor = JobSupervisor(
            agent_id=identity.id,
            store=store,
            owner=job_owner,
            runtime=capability_runtime,
            artifacts=artifact_store,
            clock=clock,
            id_factory=id_factory,
            admission_coordinator=admission_coordinator,
            distribution=distribution_owner,
            machine_scope=machine_run_scope,
            graph_model_loop=loop,
            graph_parallelism=host_capacity.graph_parallelism,
        )
        job_owner.bind_wake(job_supervisor.wake)

        async def execute_routine_run(
            occurrence: RoutineOccurrence,
            run_input: RunInput,
            observation: ResourceRevisionObservation | None,
        ) -> LoopExit | None:
            if loop is None:
                return None
            routine = await store.load_scheduled_routine(
                identity.id,
                occurrence.routine_id,
            )
            if routine is None or routine.revision != occurrence.routine_revision:
                return None
            run_input = await capability_runtime.freeze_source_scope(run_input)
            deadline = asyncio.get_running_loop().time() + limits.max_wall_time_seconds
            lease = await admission_coordinator.admit_execution(
                WorkloadClass.ROUTINE,
                run_input.id,
                conversation_id=run_input.conversation_id or run_input.id,
                absolute_deadline=deadline,
            )
            async with lease:
                (
                    conversation_exists,
                    _conversation,
                    _older_history_exists,
                ) = await store.completed_conversation_tail(
                    identity.id,
                    run_input.conversation_id or run_input.id,
                )
                if not conversation_exists:
                    raise ValueError("routine_destination_conversation_missing")
                # An approved routine is self-contained. The conversation
                # owns its inbox destination, not its reasoning context.
                prior_messages: tuple[CanonicalMessage, ...] = ()
                predecessor_run = await store.latest_terminal_conversation_run(
                    identity.id,
                    run_input.conversation_id or run_input.id,
                )
                predecessor = (
                    None
                    if predecessor_run is None
                    else ConversationPredecessor.from_run(predecessor_run)
                )
                session = RunSession(
                    run=run_input,
                    writer=RunSessionWriter(
                        store,
                        run_input,
                        predecessor=predecessor,
                    ),
                    absolute_deadline=deadline,
                    cancellation=lease.cancellation,
                    options=RunSessionOptions(
                        retained_skill_bindings=tuple(
                            (binding.skill_name, binding.content_digest)
                            for binding in routine.skill_bindings
                        ),
                        requester=routine.title,
                    ),
                    admission_lease=lease,
                    predecessor=predecessor,
                )
                prepared = await loop.prepare(
                    session,
                    prior_messages=prior_messages,
                )
                assert occurrence.claim_token is not None
                assert run_input.execution_scope is not None
                bound = await store.bind_routine_occurrence_run(
                    identity.id,
                    occurrence.occurrence_id,
                    claim_token=occurrence.claim_token,
                    run_id=run_input.id,
                    execution_scope=run_input.execution_scope,
                    bound_at=clock(),
                    precheck_observation=observation,
                )
                if bound is None:
                    return None
                try:
                    return await loop.run(
                        session.with_prepared(prepared),
                        prior_messages=prior_messages,
                    )
                except asyncio.CancelledError:
                    executing = asyncio.current_task()
                    if executing is None or executing.cancelling():
                        raise
                    # An unanswered approval interrupted the run, not the
                    # supervisor: finish the occurrence from the exit the loop
                    # recorded so it shows as Needs you.
                    return await store.result(run_input.id)

        routine_supervisor = RoutineSupervisor(
            agent_id=identity.id,
            store=store,
            owner=routine_owner,
            runtime=capability_runtime,
            distribution=distribution_owner,
            artifacts=artifact_store,
            execute_run=(execute_routine_run if loop is not None else None),
            machine_scope=machine_run_scope,
            clock=clock,
            id_factory=id_factory,
        )
        routine_owner.bind_wake(routine_supervisor.wake)
        embedded = cls(
            identity=identity,
            home=home,
            workspace=workspace,
            workspace_backend=workspace_backend,
            writer_lock=writer_lock,
            store=store,
            distribution_owner=distribution_owner,
            loop=loop,
            transcripts=transcripts,
            capabilities=capabilities,
            catalog_service=catalog_service,
            data_view=data_view,
            capability_runtime=capability_runtime,
            approval_handler=approval_handler,
            job_owner=job_owner,
            routine_owner=routine_owner,
            routine_supervisor=routine_supervisor,
            job_supervisor=job_supervisor,
            data_profile_admission=data_profile_admission,
            execution_contract_reader=read_execution_contracts,
            data_profile_job_domain=data_profile_job_domain,
            semantic_domain=semantic_domain,
            memory_store=memory_store,
            skill_store=skill_store,
            context_builder=(
                resolved_context
                if isinstance(resolved_context, AgentContextBuilder)
                else None
            ),
            artifact_store=artifact_store,
            artifact_delivery=artifact_delivery,
            admission_coordinator=admission_coordinator,
            host_capacity=host_capacity,
            source_types=registry,
            source_backends=source_backends,
            combined_query_engine=combined_engine,
            model_profile=model_profile,
            model_route=model_route,
            owned_model_provider=owned_model_provider,
            limits=limits,
            model_call_policy=model_call_policy,
            secret_provider=secret_provider,
            keychain=keychain,
            owns_credential_session=owns_credential_session,
            model_validator=model_validator,
            mcp_client_factory=resolved_mcp_client_factory,
            mcp_connection_provider=mcp_connection_provider,
            mcp_activated_bindings=mcp_activated_bindings,
            stage_mcp_catalog=stage_mcp_catalog,
            close_state=close_state,
            hosted=hosted,
            clock=clock,
            id_factory=id_factory,
        )
        if admission_cleanup is not None:
            admission_cleanup.pop_all()  # The completed runtime now owns cleanup.
        try:
            await job_supervisor.start()
            await routine_supervisor.start()
        except BaseException:
            await embedded.close()
            raise
        return embedded

    @staticmethod
    def model_requires_explicit_limits(*, provider: str, model: str) -> bool:
        """Project exact release-reviewed profile admission through the facade."""

        if not isinstance(provider, str) or not provider:
            raise ValueError("provider must be non-empty text")
        if not isinstance(model, str) or not model:
            raise ValueError("model must be non-empty text")
        return reviewed_model_profile(f"{provider}:{model}") is None

    async def authenticate_model_subscription(
        self,
        *,
        provider: str,
        on_verification: Callable[[CodexDevicePrompt], None],
        on_progress: Callable[[str], None] | None = None,
    ) -> str:
        """Run one provider-owned subscription login without persisting it."""

        self._require_open()
        definition = provider_definition(provider)
        if (
            definition is None
            or definition.authentication is not AuthenticationMode.CODEX_SUBSCRIPTION
        ):
            raise ValueError(
                "integrated subscription login is available only for Codex"
            )
        return await login_codex_subscription(
            on_verification=on_verification,
            on_progress=on_progress,
        )

    async def configure_model(
        self,
        *,
        provider: str,
        model: str,
        api_key: str | None = None,
        subscription_credential: str | None = None,
        base_url: str | None = None,
        context_window_tokens: int | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelRoute:
        """Validate and atomically persist one non-secret model route.

        The active loop is deliberately left unchanged. Callers close and reopen
        the agent before using the replacement route.
        Omitting a subscription credential reuses the current primary route's
        login only for the same subscription provider and endpoint.
        """

        provider_name, model_name, endpoint, requires_credential = (
            admit_model_selection(
                provider,
                model,
                base_url,
            )
        )
        definition = provider_definition(provider_name)
        requires_subscription_credential = (
            definition is not None
            and definition.authentication is AuthenticationMode.CODEX_SUBSCRIPTION
        )
        if requires_subscription_credential:
            if api_key is not None:
                raise ValueError("Codex subscription login does not accept an API key")
            if subscription_credential is not None and (
                not isinstance(subscription_credential, str)
                or not subscription_credential
            ):
                raise ValueError("Codex requires a Daita subscription login")
            credential_value = subscription_credential
        elif requires_credential:
            if subscription_credential is not None:
                raise ValueError("API providers do not accept a subscription login")
            if not isinstance(api_key, str) or not api_key:
                raise ValueError("API key is required for this provider")
            credential_value = api_key
        else:
            if api_key is not None or subscription_credential is not None:
                raise ValueError(
                    "this subscription or local provider does not accept a "
                    "credential during onboarding"
                )
            credential_value = None
        if (
            credential_value is not None
            and len(credential_value.encode("utf-8")) > 64 * 1_024
        ):
            raise ValueError("provider credential exceeds its 64 KiB bound")

        async with self._admit_owned_system_work(
            "model-validation", self._model_management_lock
        ):
            self._require_open()
            try:
                previous = await asyncio.to_thread(
                    _read_model_configuration,
                    self.home,
                    self.identity.id,
                )
            except AgentModelConfigurationError:
                # A recovery open deliberately has no active route. The rejected
                # profile remains untouched until validation and the atomic
                # replacement commit both succeed.
                previous = None
            reference: SecretReference | None = None
            reuse_subscription = (
                requires_subscription_credential and credential_value is None
            )
            if reuse_subscription:
                if previous is not None and previous.model_route is not None:
                    candidate = previous.model_route.candidates[0]
                    if (
                        candidate.provider_id.partition(":")[0] == provider_name
                        and candidate.base_url == endpoint
                    ):
                        reference = candidate.secret_reference
                if reference is None:
                    raise ValueError("Codex requires a Daita subscription login")
            elif requires_credential:
                reference = SecretReference.keychain(
                    _credential_account(
                        self.identity.id,
                        provider_name,
                        self._id_factory("credential"),
                    )
                )
            route = _model_route(
                provider_name,
                model_name,
                base_url=endpoint,
                secret_reference=reference,
                context_window_tokens=context_window_tokens,
                max_output_tokens=max_output_tokens,
            )
            validate_mcp_model_compatibility(
                await self._store.list_mcp_bindings(self.identity.id),
                tuple(candidate.provider_id for candidate in route.candidates),
            )
            replacement = AgentConfig(
                model_route=route,
                limits=self._limits,
                model_call_policy=self._model_call_policy,
            )
            committed = False
            try:
                if reference is not None and not reuse_subscription:
                    assert credential_value is not None
                    credential = credential_value
                    api_key = None
                    subscription_credential = None
                    try:
                        await self._keychain.set(reference, credential)
                    finally:
                        del credential
                await _validate_model_route(
                    route,
                    secret_provider=self._secret_provider or self._keychain,
                    injected_provider=self._model_validator,
                    injected_concurrency_key=(
                        None
                        if self._model_validator is None
                        else _injected_concurrency_key(
                            self._model_validator.provider_id,
                            identity=self.identity.id,
                            home=self._home,
                        )
                    ),
                    call_policy=self._model_call_policy,
                    provider_admission=self._provider_admission,
                )
                await _await_sync_completion(
                    lambda: _write_model_configuration(self.home, replacement)
                )
                committed = True
                self._model_reopen_required = True
                for provider_name, old_reference in _keychain_references(previous):
                    if old_reference != reference and _credential_reference_is_owned(
                        old_reference,
                        agent_id=self.identity.id,
                        provider=provider_name,
                    ):
                        try:
                            await asyncio.wait_for(
                                self._keychain.delete(old_reference),
                                timeout=_CREDENTIAL_CLEANUP_TIMEOUT_SECONDS,
                            )
                        except BaseException:
                            pass
                return route
            except BaseException:
                if reference is not None and not reuse_subscription and not committed:
                    try:
                        await self._keychain.delete(reference)
                    except BaseException:
                        pass
                raise

    async def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_scope_ids: tuple[str, ...] = (),
        files_only: bool = False,
        caller_principal_id: str | None = None,
    ) -> LoopExit:
        return await self._run(
            message,
            conversation_id=conversation_id,
            source_scope_ids=source_scope_ids,
            files_only=files_only,
            caller_principal_id=caller_principal_id,
        )

    async def learn(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_scope_ids: tuple[str, ...] = (),
        caller_principal_id: str | None = None,
    ) -> LoopExit:
        """Run one explicit user-authorized foreground learning action."""

        return await self._run(
            message,
            conversation_id=conversation_id,
            source_scope_ids=source_scope_ids,
            explicit_learning=True,
            caller_principal_id=caller_principal_id,
        )

    async def _run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_scope_ids: tuple[str, ...] = (),
        explicit_learning: bool = False,
        files_only: bool = False,
        run_id: str | None = None,
        evidence: RunSessionEvidence | None = None,
        one_time_artifact_destinations: tuple[ArtifactDestinationGrant, ...] = (),
        caller_principal_id: str | None = None,
    ) -> LoopExit:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        principal_id = self._resolve_caller_principal(caller_principal_id)
        if not isinstance(source_scope_ids, tuple) or any(
            not isinstance(item, str) or not item.strip() for item in source_scope_ids
        ):
            raise ValueError("source_scope_ids must be a tuple of exact source IDs")
        if not isinstance(files_only, bool):
            raise TypeError("files_only must be bool")
        if files_only and source_scope_ids:
            raise ValueError("files_only and source_scope_ids are mutually exclusive")
        if self._model_reopen_required:
            raise AgentNotConfiguredError(
                "model configuration changed; close and reopen required"
            )
        loop = self._require_loop()
        self._require_open()
        supplied_conversation = conversation_id is not None
        resolved_conversation = (
            self._id_factory("conversation")
            if conversation_id is None
            else conversation_id
        )
        _validate_conversation_id(resolved_conversation)
        session_source_id = self._data_view.session_source_id
        if files_only and session_source_id is None:
            raise AgentHomeError("files_only requires admitted local file access")
        resolved_run_id = run_id or self._id_factory("run")
        deadline = (
            asyncio.get_running_loop().time() + self._limits.max_wall_time_seconds
        )
        try:
            lease = await self._admission_coordinator.admit_execution(
                WorkloadClass.FOREGROUND,
                resolved_run_id,
                conversation_id=resolved_conversation,
                absolute_deadline=deadline,
            )
        except AdmissionClosedError as error:
            raise AgentHomeError("embedded agent is closed") from error
        async with lease:
            (
                conversation_exists,
                conversation,
                older_history_exists,
            ) = await self._store.completed_conversation_tail(
                self.identity.id,
                resolved_conversation,
            )
            if supplied_conversation and not conversation_exists:
                raise ValueError("unknown conversation for this agent")
            if not supplied_conversation and conversation_exists:
                raise ValueError("generated conversation id already exists")
            if conversation_exists and self._hosted:
                await self._require_conversation_caller(
                    resolved_conversation, principal_id
                )
            active_ids = {
                source.id
                for source in await self._store.list_sources(self.identity.id)
                if source.active
            }
            if session_source_id is not None:
                active_ids.add(session_source_id)
            if not set(source_scope_ids) <= active_ids:
                raise ValueError("A requested source is not active for this agent")
            prior_messages = _project_completed_history(
                conversation,
                older_history_exists=older_history_exists,
                capabilities=self._capabilities,
                mcp_tool_contract=self._read_mcp_tool_contract,
            )
            run_input = RunInput(
                id=resolved_run_id,
                agent_id=self.identity.id,
                caller_principal_id=principal_id,
                caller_principal_verified=(
                    not self._hosted or caller_principal_id is not None
                ),
                message=message.strip(),
                created_at=self._clock(),
                conversation_id=resolved_conversation,
                source_scope_ids=source_scope_ids,
                history_sensitivity=max(
                    (
                        item.result.sensitivity
                        for item in conversation
                        if item.result is not None
                    ),
                    key=lambda item: item.routing_rank,
                    default=ModelSensitivity.PUBLIC,
                ),
            )
            # Files-only scopes the run to exactly the live session folder source.
            run_input = replace(
                run_input,
                resolved_source_scope=(
                    EffectiveSourceScope(
                        frozenset({cast(str, session_source_id)}), frozenset()
                    )
                    if files_only
                    else await resolve_effective_source_scope(
                        run_input, self._data_view
                    )
                ),
            )
            predecessor_run = await self._store.latest_terminal_conversation_run(
                self.identity.id,
                resolved_conversation,
            )
            predecessor = (
                None
                if predecessor_run is None
                else ConversationPredecessor.from_run(predecessor_run)
            )
            options = RunSessionOptions(
                files_only=files_only,
                explicit_learning=explicit_learning,
                one_time_artifact_destinations=one_time_artifact_destinations,
            )
            session = RunSession(
                run=run_input,
                writer=RunSessionWriter(
                    self._transcripts,
                    run_input,
                    predecessor=predecessor,
                ),
                absolute_deadline=deadline,
                cancellation=lease.cancellation,
                options=options,
                admission_lease=lease,
                predecessor=predecessor,
                evidence=evidence or RunSessionEvidence(),
            )
            return await loop.run(
                session,
                prior_messages=prior_messages,
            )

    async def transcript(
        self, run_id: str, *, caller_principal_id: str | None = None
    ) -> Transcript:
        self._require_open()
        principal_id = self._resolve_caller_principal(caller_principal_id)
        transcript = await self._transcripts.load(run_id)
        if self._hosted and transcript.run.caller_principal_id != principal_id:
            raise ValueError("run is unavailable to this caller")
        return transcript

    async def tool_evidence(
        self, run_id: str, *, caller_principal_id: str | None = None
    ) -> tuple[tuple[ToolCall, ToolResultBlock | None], ...]:
        transcript = await self.transcript(
            run_id, caller_principal_id=caller_principal_id
        )
        return self._capabilities.executed_tool_pairs(transcript)

    async def analysis_evidence(
        self, run_id: str, *, caller_principal_id: str | None = None
    ):
        await self.transcript(run_id, caller_principal_id=caller_principal_id)
        return await self._store.load_analysis_evidence(self.identity.id, run_id)

    def postgresql_read_diagnostics(self) -> PostgreSQLReadDiagnostics:
        backend = self._source_backends[SourceOperation.SQL_READ][POSTGRESQL_ADAPTER_ID]
        return cast(PostgreSQLReadDiagnostics, backend.diagnostics())

    def admission_diagnostics(self) -> CoordinatorDiagnostics:
        return self._admission_coordinator.diagnostics()

    def analysis_runtime(self) -> dict[str, object]:
        self._require_open()
        if self._hosted:
            return {"available": False, "reason": "hosted_analysis_not_admitted"}
        from ..adapters.analytical_workspace.runtime import runtime_status

        return runtime_status()

    async def analysis_usage(
        self, run_id: str, *, caller_principal_id: str | None = None
    ) -> dict[str, object] | None:
        from ..loop.analysis import analysis_summary

        return analysis_summary(
            await self.analysis_evidence(
                run_id, caller_principal_id=caller_principal_id
            )
        )

    async def _require_conversation_caller(
        self, conversation_id: str, principal_id: str
    ) -> bool:
        exists, allowed = await self._store.conversation_access(
            self.identity.id,
            conversation_id,
            caller_principal_id=principal_id if self._hosted else None,
        )
        if exists and not allowed:
            raise ValueError("conversation is unavailable to this caller")
        return exists

    async def list_deliveries(
        self,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = 50,
        caller_principal_id: str | None = None,
    ) -> tuple[DeliveryView, ...]:
        """List bounded agent-owned durable deliveries."""

        self._require_open()
        principal_id = self._resolve_caller_principal(caller_principal_id)
        if conversation_id is not None:
            _validate_conversation_id(conversation_id)
            await self._require_conversation_caller(conversation_id, principal_id)
        elif self._hosted:
            raise ValueError("hosted delivery listing requires an owned conversation")
        items = await self._distribution_owner.list(
            conversation_id=conversation_id,
            include_acknowledged=include_acknowledged,
            limit=limit,
        )
        return items

    async def distribution_destinations(
        self,
        conversation_id: str,
        *,
        sensitivity_ceiling: ModelSensitivity,
        caller_principal_id: str | None = None,
    ) -> tuple[DistributionDestination, ...]:
        """Discover exact current destinations for one owned conversation."""

        self._require_open()
        _validate_conversation_id(conversation_id)
        await self._require_conversation_caller(
            conversation_id, self._resolve_caller_principal(caller_principal_id)
        )
        if not isinstance(sensitivity_ceiling, ModelSensitivity):
            raise TypeError("sensitivity_ceiling must be ModelSensitivity")
        return await self._distribution_owner.discover_destinations(
            conversation_id,
            sensitivity_ceiling=sensitivity_ceiling,
        )

    async def inspect_delivery(
        self,
        delivery_id: str,
        *,
        caller_principal_id: str | None = None,
    ) -> DeliveryInspection | None:
        """Inspect one exact agent-owned logical delivery."""

        self._require_open()
        if not isinstance(delivery_id, str) or not delivery_id.strip():
            raise ValueError("delivery_id must be non-empty text")
        principal_id = self._resolve_caller_principal(caller_principal_id)
        result = await self._distribution_owner.inspect(delivery_id.strip())
        if result is not None:
            await self._require_conversation_caller(
                result.delivery.conversation_id, principal_id
            )
        return result

    async def acknowledge_delivery(
        self, delivery_id: str, *, caller_principal_id: str | None = None
    ) -> DeliveryView | None:
        """Idempotently acknowledge one exact agent-owned delivery."""

        self._require_open()
        if not isinstance(delivery_id, str) or not delivery_id.strip():
            raise ValueError("delivery_id must be non-empty text")
        principal_id = self._resolve_caller_principal(caller_principal_id)
        inspection = await self._distribution_owner.inspect(delivery_id.strip())
        if inspection is not None:
            await self._require_conversation_caller(
                inspection.delivery.conversation_id, principal_id
            )
        acknowledged = await self._distribution_owner.acknowledge(
            delivery_id.strip(),
            acknowledged_at=self._clock(),
        )
        if acknowledged is not None:
            self._routine_supervisor.wake()
        return acknowledged

    async def work_board(
        self,
        *,
        done_window: timedelta = DEFAULT_DONE_WINDOW,
        caller_principal_id: str | None = None,
    ) -> WorkBoard:
        """Project this agent's live and recent jobs, routines and results."""

        self._require_open()
        self._resolve_caller_principal(caller_principal_id)
        if self._hosted:
            raise ValueError("hosted work board is not available yet")
        if not isinstance(done_window, timedelta) or not (
            timedelta(0) < done_window <= MAX_DONE_WINDOW
        ):
            raise ValueError("done window must be positive and at most seven days")
        now = self._clock()
        live_jobs = await self._job_owner.list(states=_LIVE_JOB_STATES, limit=100)
        finished_jobs = await self._job_owner.list(
            states=TERMINAL_GRAPH_STATES, limit=100
        )
        routines = await self._routine_owner.list(limit=MAX_ROUTINE_LIST_PAGE_SIZE)
        active = await self._routine_owner.list_active_occurrences(limit=100)
        deliveries = await self._distribution_owner.list()
        results = await self._routine_owner.load_occurrences(
            tuple(
                dict.fromkeys(
                    item.subject_id
                    for item in deliveries
                    if item.subject_kind is DeliverySubjectKind.ROUTINE_OCCURRENCE
                )
            )
        )
        return project_work_board(
            jobs=(*live_jobs, *finished_jobs),
            routines=routines,
            active_occurrences=active,
            result_occurrences=results,
            deliveries=deliveries,
            now=now,
            done_window=done_window,
        )

    def background_status(self) -> tuple[SupervisorStatus, ...]:
        """Return payload-free health of background job and routine execution."""
        return (self._job_supervisor.status, self._routine_supervisor.status)

    async def conversation_run_page(
        self,
        conversation_id: str,
        *,
        after_turn_index: int = -1,
        limit: int = 100,
        caller_principal_id: str | None = None,
    ) -> tuple[ConversationRun, ...]:
        self._require_open()
        _validate_conversation_id(conversation_id)
        principal = self._resolve_caller_principal(caller_principal_id)
        if not await self._require_conversation_caller(conversation_id, principal):
            raise ValueError("unknown conversation for this agent")
        records = await self._store.conversation_run_page(
            self.identity.id,
            conversation_id,
            after_turn_index=after_turn_index,
            limit=limit,
        )
        if self._hosted and any(
            item.transcript.run.caller_principal_id != principal for item in records
        ):
            raise ValueError("conversation is unavailable to this caller")
        return records

    async def conversation_runs(
        self,
        conversation_id: str,
        *,
        caller_principal_id: str | None = None,
    ) -> tuple[ConversationRun, ...]:
        self._require_open()
        _validate_conversation_id(conversation_id)
        records = await self._store.conversation_runs(
            self.identity.id,
            conversation_id,
        )
        if not records:
            raise ValueError("unknown conversation for this agent")
        principal_id = self._resolve_caller_principal(caller_principal_id)
        if self._hosted and any(
            item.transcript.run.caller_principal_id != principal_id for item in records
        ):
            raise ValueError("conversation is unavailable to this caller")
        return records

    async def conversation_exists(
        self, conversation_id: str, *, caller_principal_id: str | None = None
    ) -> bool:
        """Validate and check one bounded agent-scoped conversation identity."""

        self._require_open()
        _validate_conversation_id(conversation_id)
        principal_id = self._resolve_caller_principal(caller_principal_id)
        try:
            return await self._require_conversation_caller(
                conversation_id, principal_id
            )
        except ValueError:
            return False

    async def list_jobs(
        self,
        *,
        states: frozenset[GraphState] = frozenset(),
        limit: int = 50,
    ) -> tuple[GraphJob, ...]:
        """Return the bounded current projection of this agent's durable jobs."""

        self._require_open()
        return await self._job_owner.list(states=states, limit=limit)

    async def inspect_effect(
        self, receipt_id: str, *, caller_principal_id: str | None = None
    ) -> EffectReceipt | None:
        """Read one exact bounded agent-owned observation and separate resolution."""
        self._require_open()
        validate_effect_receipt_id(receipt_id)
        principal_id = self._resolve_caller_principal(caller_principal_id)
        receipt = await self._store.load_effect_receipt(self.identity.id, receipt_id)
        if self._hosted and receipt is not None:
            transcript = await self._transcripts.load(receipt.run_id)
            if transcript.run.caller_principal_id != principal_id:
                return None
        return receipt

    async def list_effects(
        self,
        *,
        unresolved_only: bool = False,
        limit: int = 20,
        offset: int = 0,
        caller_principal_id: str | None = None,
    ) -> tuple[EffectReceipt, ...]:
        self._require_open()
        principal_id = self._resolve_caller_principal(caller_principal_id)
        return await self._store.list_effect_receipts(
            self.identity.id,
            unresolved_only=unresolved_only,
            limit=limit,
            offset=offset,
            caller_principal_id=principal_id if self._hosted else None,
        )

    async def resolve_effect(
        self,
        receipt_id: str,
        *,
        expected_digest: str,
        decision: EffectResolutionDecision,
        note: str,
        evidence_references: tuple[str, ...] = (),
        caller_principal_id: str | None = None,
    ) -> EffectReceipt:
        """Approve one exact human recovery decision; never retry an operation."""
        async with self._admit_system_work("effect-recovery"):
            self._require_open()
            principal_id = self._resolve_caller_principal(caller_principal_id)
            receipt = await self.inspect_effect(
                receipt_id, caller_principal_id=principal_id
            )
            if receipt is None or receipt.receipt_digest != expected_digest:
                raise ValueError(
                    "the exact owned receipt or expected observation digest is unavailable"
                )
            resolution = EffectResolution(
                receipt_id=receipt_id,
                receipt_digest=expected_digest,
                decision=decision,
                approving_principal_id=principal_id,
                control_id=self._id_factory("effect-recovery"),
                resolved_at=max(
                    self._clock(), receipt.finished_at or receipt.started_at
                ),
                note=note,
                evidence_references=evidence_references,
            )
            if receipt.resolution is not None:
                prior = receipt.resolution
                if (
                    prior.decision is decision
                    and prior.note == note
                    and prior.evidence_references == resolution.evidence_references
                ):
                    return receipt
                raise ValueError(
                    "this receipt already has a different immutable recovery decision"
                )
            if receipt.outcome is not EffectOutcome.UNCERTAIN:
                raise ValueError("only terminal uncertainty requires explicit recovery")
            if self._approval_handler is None:
                raise PermissionError(
                    "effect recovery requires the foreground approval handler"
                )
            evidence = await self._effect_recovery_evidence(
                resolution.evidence_references,
                caller_principal_id=principal_id,
            )
            request = ApprovalRequest(
                run_id=resolution.control_id,
                call_id=resolution.control_id,
                tool_name="resolve_effect",
                capability_id="control.resolve_effect",
                arguments=FrozenJsonObject.from_mapping(
                    {
                        "receipt": receipt.material(),
                        "receipt_digest": receipt.receipt_digest,
                        "decision": decision.value,
                        "user_note": note,
                        "evidence_references": evidence,
                        "consequence": (
                            "Disable the producing routine without retrying the operation."
                            if decision is EffectResolutionDecision.CLOSE_WITHOUT_RETRY
                            else "Accept possible duplication in future authorized work; leave the producing routine paused until explicit resume or run-now."
                        ),
                        "observation_unchanged": True,
                    }
                ),
                reason="Approve this exact recovery decision? The original external outcome remains uncertain; recovery grants no connector authority and performs no retry.",
            )
            if request.render_arguments_for_review() is None:
                raise ValueError(
                    "the exact recovery document exceeds the approval display bound"
                )
            if await self._approval_handler(request) is not ApprovalDecision.APPROVE:
                raise PermissionError("the effect recovery decision was denied")
            async with self._effect_resolution_lock:
                self._require_open()
                current = await self.inspect_effect(
                    receipt_id, caller_principal_id=principal_id
                )
                if (
                    current != receipt
                    or await self._effect_recovery_evidence(
                        resolution.evidence_references,
                        caller_principal_id=principal_id,
                    )
                    != evidence
                ):
                    raise ValueError(
                        "recovery evidence changed while approval was pending"
                    )
                return await self._store.resolve_effect_receipt(
                    self.identity.id, resolution
                )

    async def _effect_recovery_evidence(
        self,
        references: tuple[str, ...],
        *,
        caller_principal_id: str | None = None,
    ) -> tuple[dict[str, object], ...]:
        evidence: list[dict[str, object]] = []
        for reference in references:
            if reference.startswith("effect-receipt:"):
                receipt = await self.inspect_effect(
                    reference, caller_principal_id=caller_principal_id
                )
                if receipt is None or receipt.outcome is EffectOutcome.STARTED:
                    raise ValueError(
                        "recovery evidence requires exact owned terminal receipts"
                    )
                evidence.append(
                    {
                        "receipt_id": receipt.receipt_id,
                        "receipt_digest": receipt.receipt_digest,
                    }
                )
            else:
                await self._require_artifact_caller(reference, caller_principal_id)
                artifact = await self._artifact_store.find_ref(reference)
                evidence.append(
                    {"artifact_id": artifact.artifact_id, "sha256": artifact.sha256}
                )
        return tuple(evidence)

    async def inspect_job(self, job_id: str) -> GraphInspection | None:
        self._require_open()
        return await self._job_owner.inspect(job_id)

    async def list_job_tasks(
        self,
        job_id: str,
        *,
        states: frozenset[TaskState] = frozenset(),
        limit: int = 64,
    ) -> tuple[GraphTask, ...]:
        self._require_open()
        return await self._job_owner.list_graph_tasks(
            job_id, states=states, limit=limit
        )

    async def inspect_job_task(self, job_id: str, task_id: str) -> GraphTask | None:
        self._require_open()
        return await self._job_owner.inspect_graph_task(job_id, task_id)

    async def list_job_dependencies(
        self, job_id: str, *, task_id: str | None = None, limit: int = 100
    ) -> tuple[TaskDependency, ...]:
        self._require_open()
        return await self._job_owner.list_graph_dependencies(
            job_id, task_id=task_id, limit=limit
        )

    async def list_task_attempts(
        self, job_id: str, task_id: str, *, limit: int = 3
    ) -> tuple[TaskAttempt, ...]:
        self._require_open()
        return await self._job_owner.list_graph_task_attempts(
            job_id, task_id, limit=limit
        )

    async def read_task_result(self, job_id: str, task_id: str) -> TaskResult | None:
        self._require_open()
        return await self._job_owner.read_graph_task_result(job_id, task_id)

    async def list_task_checkpoints(
        self, job_id: str, task_id: str, *, limit: int = 8
    ) -> tuple[TaskCheckpoint, ...]:
        self._require_open()
        return await self._job_owner.list_graph_task_checkpoints(
            job_id, task_id, limit=limit
        )

    async def list_task_artifacts(
        self, job_id: str, *, task_id: str | None = None, limit: int = 64
    ) -> tuple[str, ...]:
        self._require_open()
        artifact_ids = await self._job_owner.list_graph_task_artifacts(
            job_id, task_id=task_id, limit=limit
        )
        return tuple(
            [
                artifact_id
                for artifact_id in artifact_ids
                if (record := await self._store.get_artifact_record(artifact_id))
                is not None
                and record.state is ArtifactState.READY
            ]
        )

    async def list_task_controls(
        self, job_id: str, task_id: str, *, limit: int = 8
    ) -> tuple[TaskControl, ...]:
        self._require_open()
        return await self._job_owner.list_graph_task_controls(
            job_id, task_id, limit=limit
        )

    async def job_timeline(
        self,
        job_id: str,
        *,
        after_event_id: int = 0,
        limit: int = 100,
        task_id: str | None = None,
    ) -> GraphTimelinePage | None:
        self._require_open()
        return await self._job_owner.graph_timeline(
            job_id,
            after_event_id=after_event_id,
            limit=limit,
            task_id=task_id,
        )

    async def job_board(self, job_id: str) -> GraphBoardProjection | None:
        self._require_open()
        return await self._job_owner.graph_board(job_id)

    async def read_job_result(self, job_id: str) -> TaskResult | None:
        self._require_open()
        return await self._job_owner.read_result(job_id)

    async def cancel_job(self, job_id: str) -> GraphJob | None:
        self._require_open()
        return await self._job_owner.cancel(job_id)

    async def graph_blockers(self, job_id: str) -> GraphBlockerProjection | None:
        self._require_open()
        return await self._job_owner.graph_blockers(job_id)

    async def answer_task_input(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        answer: Mapping[str, object],
        idempotency_key: str | None = None,
    ) -> TaskControl | None:
        self._require_open()
        return await self._job_owner.answer_graph_task_input(
            job_id,
            task_id,
            control_id,
            principal_id=principal_id,
            answer=answer,
            idempotency_key=idempotency_key,
        )

    async def accept_task_review(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        rationale: str,
        idempotency_key: str,
    ) -> TaskResult:
        self._require_open()
        return await self._job_owner.accept_graph_task_review_by_principal(
            job_id,
            task_id,
            control_id,
            principal_id=principal_id,
            rationale=rationale,
            idempotency_key=idempotency_key,
        )

    async def request_task_review_changes(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        rationale: str,
        replacement_guidance: str,
        idempotency_key: str,
    ) -> TaskControl:
        self._require_open()
        return await self._job_owner.request_graph_task_review_changes_by_principal(
            job_id,
            task_id,
            control_id,
            principal_id=principal_id,
            rationale=rationale,
            replacement_guidance=replacement_guidance,
            idempotency_key=idempotency_key,
        )

    async def retry_task_control(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        advisory_note: str,
        idempotency_key: str,
    ) -> TaskControl | None:
        self._require_open()
        return await self._job_owner.retry_graph_task_control(
            job_id,
            task_id,
            control_id,
            principal_id=principal_id,
            advisory_note=advisory_note,
            idempotency_key=idempotency_key,
        )

    async def reject_task_control(
        self,
        job_id: str,
        task_id: str,
        control_id: str,
        *,
        principal_id: str,
        reason: str,
        idempotency_key: str | None = None,
    ) -> TaskControl | None:
        self._require_open()
        return await self._job_owner.reject_graph_task_control(
            job_id,
            task_id,
            control_id,
            principal_id=principal_id,
            reason=reason,
            idempotency_key=idempotency_key,
        )

    async def cancel_graph_job(
        self, job_id: str, *, principal_id: str
    ) -> GraphJob | None:
        self._require_open()
        return await self._job_owner.cancel_graph_job(job_id, principal_id=principal_id)

    async def replace_graph_task_by_policy(
        self,
        job_id: str,
        task_id: str,
        *,
        principal_id: str,
        advisory_note: str,
        idempotency_key: str,
        expected_revision: int,
    ) -> GraphMutation:
        self._require_open()
        return await self._job_owner.replace_graph_task_by_policy(
            job_id,
            task_id,
            principal_id=principal_id,
            advisory_note=advisory_note,
            idempotency_key=idempotency_key,
            expected_revision=expected_revision,
        )

    async def start_authorized_replacement_job(
        self,
        admission: GraphAdmission,
        *,
        replaces_job_id: str,
        principal_id: str,
        replaces_task_id: str | None = None,
        control_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> GraphJob:
        self._require_open()
        return await self._job_owner.admit_authorized_replacement_graph(
            admission,
            replaces_job_id=replaces_job_id,
            principal_id=principal_id,
            replaces_task_id=replaces_task_id,
            control_id=control_id,
            idempotency_key=idempotency_key,
        )

    async def propose_routine(self, draft: ScheduledRoutineDraft) -> ScheduledRoutine:
        """Build and revalidate one non-persisted exact routine proposal."""

        self._require_open()
        return await self._prepare_routine_proposal(draft, basis_run_id=None)

    async def promote_routine(
        self,
        draft: ScheduledRoutineDraft,
        *,
        basis_run_id: str,
    ) -> ScheduledRoutine:
        """Build a proposal with exact evidence from one completed prior run."""

        self._require_open()
        return await self._prepare_routine_proposal(
            draft,
            basis_run_id=basis_run_id,
        )

    async def create_routine(
        self,
        proposal: ScheduledRoutine,
        *,
        confirmation_handler: ApprovalHandler | None = None,
    ) -> ScheduledRoutine:
        """Persist one exact proposal, presenting it to an explicit human confirmation handler."""

        self._require_open()
        if proposal.agent_id != self.identity.id:
            raise ValueError("routine proposal belongs to another agent")
        authority = await self._routine_owner.proposal_authority_snapshot(proposal)
        if confirmation_handler is not None:
            control_id = self._id_factory("routine-approval")
            request = ApprovalRequest(
                run_id=control_id,
                call_id=control_id,
                tool_name=ROUTINE_SAVE_TOOL_NAME,
                capability_id="routines.create",
                arguments=routine_approval_arguments(authority),
                reason="Approve this exact saved assignment, its standing actions, and its schedule?",
            )
            if (
                request.render_arguments_for_review() is None
                or await confirmation_handler(request) is not ApprovalDecision.APPROVE
            ):
                raise PermissionError("the routine creation was not approved")
        async with self._routine_management_lock:
            self._require_open()
            if (
                await self._routine_owner.proposal_authority_snapshot(proposal)
                != authority
            ):
                raise ValueError(
                    "routine authority changed during approval; propose again"
                )
            return await self._routine_owner.admit(proposal)

    async def list_routines(
        self,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = 50,
    ) -> tuple[ScheduledRoutineSummary, ...]:
        self._require_open()
        return await self._routine_owner.list(states=states, limit=limit)

    async def inspect_routine(
        self, routine_id: str
    ) -> ScheduledRoutineInspection | None:
        self._require_open()
        return await self._routine_owner.inspect(routine_id)

    async def update_routine(
        self,
        routine_id: str,
        *,
        expected_revision: int,
        draft: ScheduledRoutineDraft,
        basis_run_id: str | None = None,
        confirmation_handler: ApprovalHandler | None = None,
    ) -> ScheduledRoutine:
        """Prepare, review and conditionally replace one exact routine revision."""

        self._require_open()
        inspection = await self._routine_owner.inspect(routine_id)
        if inspection is None or inspection.routine.revision != expected_revision:
            raise RoutineError(
                "routine_revision_changed",
                "The routine changed or is no longer owned by this agent.",
            )
        proposal = await self._prepare_routine_revision(
            inspection.routine, draft, basis_run_id=basis_run_id
        )
        authority = await self._routine_owner.proposal_authority_snapshot(proposal)
        if confirmation_handler is not None:
            control_id = self._id_factory("routine-approval")
            request = ApprovalRequest(
                run_id=control_id,
                call_id=control_id,
                tool_name=ROUTINE_SAVE_TOOL_NAME,
                capability_id="routines.update",
                arguments=routine_approval_arguments(authority),
                reason="Replace this saved assignment with the exact proposed revision?",
            )
            if (
                request.render_arguments_for_review() is None
                or await confirmation_handler(request) is not ApprovalDecision.APPROVE
            ):
                raise PermissionError("the routine revision was not approved")
        async with self._routine_management_lock:
            self._require_open()
            if (
                await self._routine_owner.proposal_authority_snapshot(proposal)
                != authority
            ):
                raise ValueError(
                    "routine authority changed during approval; propose again"
                )
            return await self._routine_owner.revise(
                proposal, expected_revision=expected_revision
            )

    async def pause_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._control_routine(
            routine_id,
            expected_revision=expected_revision,
            action=RoutineControlAction.PAUSE,
        )

    async def resume_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._control_routine(
            routine_id,
            expected_revision=expected_revision,
            action=RoutineControlAction.RESUME,
        )

    async def run_routine_now(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._control_routine(
            routine_id,
            expected_revision=expected_revision,
            action=RoutineControlAction.RUN_NOW,
        )

    async def disable_routine(
        self, routine_id: str, *, expected_revision: int
    ) -> ScheduledRoutine:
        return await self._control_routine(
            routine_id,
            expected_revision=expected_revision,
            action=RoutineControlAction.DISABLE,
        )

    async def _prepare_routine_proposal(
        self,
        draft: ScheduledRoutineDraft,
        *,
        basis_run_id: str | None,
    ) -> ScheduledRoutine:
        if not isinstance(draft, ScheduledRoutineDraft):
            raise TypeError("draft must be ScheduledRoutineDraft")
        transcript = await self._store.load(draft.origin_run_id)
        if transcript.run.conversation_id is None:
            raise ValueError("routine origin run has no conversation")
        self._validate_routine_draft_distribution(
            draft,
            transcript.run.conversation_id,
        )
        return await self._routine_owner.prepare_create(
            run_id=draft.origin_run_id,
            conversation_id=transcript.run.conversation_id,
            call_id=self._id_factory("routine-proposal"),
            basis_run_id=basis_run_id,
            **_routine_draft_values(draft),
        )

    async def _prepare_routine_revision(
        self,
        current: ScheduledRoutine,
        draft: ScheduledRoutineDraft,
        *,
        basis_run_id: str | None,
    ) -> ScheduledRoutine:
        if not isinstance(draft, ScheduledRoutineDraft):
            raise TypeError("draft must be ScheduledRoutineDraft")
        transcript = await self._store.load(draft.origin_run_id)
        if transcript.run.conversation_id is None:
            raise ValueError("routine origin run has no conversation")
        self._validate_routine_draft_distribution(
            draft,
            transcript.run.conversation_id,
        )
        return await self._routine_owner.prepare_revision(
            current,
            run_id=draft.origin_run_id,
            conversation_id=transcript.run.conversation_id,
            basis_run_id=basis_run_id,
            **_routine_draft_values(draft),
        )

    def _validate_routine_draft_distribution(
        self,
        draft: ScheduledRoutineDraft,
        conversation_id: str,
    ) -> None:
        try:
            self._distribution_owner.resolve_plan(
                conversation_id,
                destination_id=draft.distribution_destination_id,
                sensitivity_ceiling=draft.sensitivity_ceiling,
            )
        except ValueError as error:
            raise ValueError(
                "routine distribution destination is not currently selectable"
            ) from error

    async def _control_routine(
        self,
        routine_id: str,
        *,
        expected_revision: int,
        action: RoutineControlAction,
    ) -> ScheduledRoutine:
        async with self._routine_management_lock:
            self._require_open()
            return await self._routine_owner.control(
                routine_id,
                expected_revision=expected_revision,
                action=action,
                authorized_control_call_id=self._id_factory("routine-control"),
            )

    async def clear_conversations(self) -> int:
        """Delete transcripts, not approved knowledge."""

        if self._hosted:
            raise ValueError(
                "hosted callers cannot clear agent-wide conversation history"
            )

        deadline = (
            asyncio.get_running_loop().time() + self._limits.max_wall_time_seconds
        )
        lease = await self._admission_coordinator.admit_execution(
            WorkloadClass.SYSTEM,
            self._id_factory("conversation-clear"),
            absolute_deadline=deadline,
        )
        async with lease:
            self._require_open()
            return await self._store.clear_conversations(self.identity.id)

    async def list_artifacts(
        self,
        *,
        limit: int,
        offset: int,
        caller_principal_id: str | None = None,
        newest_first: bool = False,
    ) -> tuple[ArtifactRef, ...]:
        async with self._artifact_publication_lock:
            self._require_open()
            principal_id = self._resolve_caller_principal(caller_principal_id)
            return await self._artifact_store.list_refs(
                caller_principal_id=principal_id if self._hosted else None,
                limit=limit,
                offset=offset,
                newest_first=newest_first,
            )

    async def read_artifact(
        self, artifact_id: str, *, caller_principal_id: str | None = None
    ) -> ArtifactPayload:
        async with self._artifact_publication_lock:
            self._require_open()
            await self._require_artifact_caller(artifact_id, caller_principal_id)
            return await self._artifact_store.read(artifact_id)

    async def delete_artifact(
        self, artifact_id: str, *, caller_principal_id: str | None = None
    ) -> bool:
        async with self._artifact_publication_lock:
            self._require_open()
            principal_id = self._resolve_caller_principal(caller_principal_id)
            record = await self._store.get_artifact_record(artifact_id)
            if record is not None and (
                record.agent_id != self.identity.id
                or (self._hosted and record.caller_principal_id != principal_id)
            ):
                raise ValueError("artifact is unavailable to this caller")
            return await self._artifact_store.delete(artifact_id)

    async def artifact_computation_evidence(
        self, artifact_id: str, *, caller_principal_id: str | None = None
    ) -> FrozenJsonObject:
        async with self._artifact_publication_lock:
            self._require_open()
            await self._require_artifact_caller(artifact_id, caller_principal_id)
            record = await self._store.get_artifact_record(artifact_id)
            if record is None or record.state is not ArtifactState.READY:
                raise ValueError("artifact is unavailable to this caller")
            return FrozenJsonObject.from_mapping(record.computation_evidence)

    async def _require_artifact_caller(
        self, artifact_id: str, caller_principal_id: str | None
    ) -> None:
        if not self._hosted:
            return
        principal_id = self._resolve_caller_principal(caller_principal_id)
        ref = await self._artifact_store.find_ref(artifact_id)
        record = await self._store.get_artifact_record(ref.artifact_id)
        if record is None or record.caller_principal_id != principal_id:
            raise ValueError("artifact is unavailable to this caller")

    async def save_artifact(
        self,
        artifact_id: str,
        destination: Path | None = None,
        *,
        filename: str | None = None,
        caller_principal_id: str | None = None,
    ) -> ArtifactDeliveryReceipt:
        async with self._artifact_publication_lock:
            self._require_open()
            await self._require_artifact_caller(artifact_id, caller_principal_id)
            return await self._require_artifact_delivery().save_public(
                artifact_id,
                destination=destination,
                filename=filename,
            )

    async def export_destination(self) -> ArtifactDestination:
        async with self._artifact_publication_lock:
            self._require_open()
            return await self._require_artifact_delivery().export_destination()

    async def set_export_destination(
        self,
        directory: Path,
    ) -> ArtifactDestination:
        async with self._artifact_publication_lock:
            self._require_open()
            return await self._require_artifact_delivery().set_export_destination(
                directory
            )

    async def reset_export_destination(self) -> ArtifactDestination:
        async with self._artifact_publication_lock:
            self._require_open()
            return await self._require_artifact_delivery().reset_export_destination()

    async def resolve_source(self, selector: str) -> SourceRegistration:
        """Resolve one exact ID, display name, or stable display-name alias."""

        self._require_open()
        sources = tuple(
            source
            for source in await self._store.list_sources(self.identity.id)
            if source.active
        )
        return _resolve_source_selector(selector, sources)

    async def read_memory(self) -> str:
        self._require_open()
        return await self._memory_store.read_memory()

    async def set_memory(
        self, text: str, *, sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED
    ) -> None:
        self._require_open()
        await self._memory_store.set_memory(text, sensitivity=sensitivity)

    async def read_user_profile(self) -> str:
        self._require_open()
        return await self._memory_store.read_user_profile()

    async def set_user_profile(
        self, text: str, *, sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED
    ) -> None:
        self._require_open()
        await self._memory_store.set_user_profile(text, sensitivity=sensitivity)

    async def list_semantic_annotations(
        self,
        *,
        source_id: str | None = None,
        resource_id: str | None = None,
        kind: SemanticKind | None = None,
        state: SemanticAnnotationState | None = None,
    ) -> tuple[SemanticAnnotationView, ...]:
        self._require_open()
        for value, field_name in (
            (source_id, "source_id"),
            (resource_id, "resource_id"),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{field_name} must be non-empty text or None")
        if kind is not None and not isinstance(kind, SemanticKind):
            raise TypeError("kind must be SemanticKind or None")
        if state is not None and not isinstance(state, SemanticAnnotationState):
            raise TypeError("state must be SemanticAnnotationState or None")
        views = await self._semantic_views()
        return tuple(
            view
            for view in views
            if (source_id is None or source_id in view.annotation.subject.source_ids)
            and (
                resource_id is None
                or resource_id in view.annotation.subject.resource_ids
            )
            and (kind is None or view.annotation.kind is kind)
            and (state is None or view.state is state)
        )

    async def read_semantic_annotation(
        self,
        annotation_id: str,
    ) -> SemanticAnnotationView | None:
        self._require_open()
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("annotation_id must be non-empty text")
        return next(
            (
                view
                for view in await self._semantic_views()
                if view.annotation.id == annotation_id
            ),
            None,
        )

    async def save_semantic_annotation(
        self,
        annotation: SemanticAnnotation,
        *,
        expected_sha256: str | None = None,
    ) -> bool:
        if not isinstance(annotation, SemanticAnnotation):
            raise TypeError("annotation must be SemanticAnnotation")
        async with self._semantic_management_lock:
            self._require_open()
            if annotation.agent_id != self.identity.id:
                raise ValueError("semantic annotation belongs to another agent")
            await self._semantic_domain.validate_annotation(
                self.identity.id,
                annotation,
            )
            return await self._store.save_semantic_annotation(
                self.identity.id,
                annotation,
                expected_sha256=expected_sha256,
            )

    async def delete_semantic_annotation(
        self,
        annotation_id: str,
        *,
        expected_sha256: str,
    ) -> bool:
        async with self._semantic_management_lock:
            self._require_open()
            return await self._store.delete_semantic_annotation(
                self.identity.id,
                annotation_id,
                expected_sha256=expected_sha256,
            )

    async def _semantic_views(self) -> tuple[SemanticAnnotationView, ...]:
        annotations = await self._store.list_semantic_annotations(self.identity.id)
        resource_ids = tuple(
            sorted(
                {
                    resource_id
                    for annotation in annotations
                    for resource_id in annotation.subject.resource_ids
                }
            )
        )
        facts = await self._data_view.semantic_resource_facts(
            self.identity.id,
            resource_ids,
        )
        return inspect_semantic_annotations(annotations, facts)

    async def list_skills(self) -> tuple[SkillSummary, ...]:
        self._require_open()
        return await self._skill_store.list_skills()

    async def read_skill(self, name: str) -> Skill | None:
        self._require_open()
        return await self._skill_store.read_skill(name)

    async def save_skill(
        self,
        name: str,
        description: str,
        instructions: str,
        *,
        sensitivity: ModelSensitivity = ModelSensitivity.RESTRICTED,
    ) -> bool:
        self._require_open()
        return await self._skill_store.save_skill(
            name, description, instructions, sensitivity=sensitivity
        )

    async def delete_skill(self, name: str) -> bool:
        self._require_open()
        return await self._skill_store.delete_skill(name)

    def _read_mcp_tool_contract(
        self, capability_id: str
    ) -> tuple[MCPServerBinding, MCPToolBinding] | None:
        """Inspect currently activated immutable MCP contracts without I/O."""

        for activated in self._mcp_activated_bindings.values():
            binding = activated.binding
            for tool in binding.tools:
                if tool.capability_id == capability_id:
                    return binding, tool
        return None

    async def store_mcp_bearer(self, credential: str) -> SecretReference:
        """Store a masked local setup credential under this agent's ownership."""
        if self._hosted:
            raise AgentHomeError("hosted MCP credentials require secret references")
        if (
            not isinstance(credential, str)
            or not credential
            or len(credential.encode()) > 64 * 1024
        ):
            raise ValueError("MCP credential must be non-empty and at most 64 KiB")
        async with self._credential_management_lock:
            self._require_open()
            reference = SecretReference.keychain(
                _credential_account(
                    self.identity.id, "mcp", self._id_factory("credential")
                )
            )
            try:
                await self._keychain.set(reference, credential)
            except BaseException:
                await self._keychain.delete(reference)
                raise
            finally:
                credential = ""
            return reference

    async def delete_mcp_bearer(self, reference: SecretReference) -> None:
        """Remove an unused credential created for this agent's local setup."""
        if not isinstance(
            reference, SecretReference
        ) or not _credential_reference_is_owned(
            reference, agent_id=self.identity.id, provider="mcp"
        ):
            raise ValueError("credential does not belong to this agent's MCP setup")
        async with self._admit_owned_system_work(
            "mcp-credential-delete", self._credential_management_lock
        ):
            if any(
                item.authentication.secret_reference == reference
                for item in await self._store.list_mcp_bindings(self.identity.id)
            ):
                raise ValueError("credential is retained by an MCP binding")
            self._require_open()
            await self._keychain.delete(reference)

    async def inspect_mcp_server(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication | None = None,
        caller_principal_id: str | None = None,
    ) -> MCPServerInspection:
        """Inspect one exact endpoint without persisting execution authority."""

        async with self._admit_system_work("mcp-inspect"):
            self._require_open()
            if (
                self._hosted
                and caller_principal_id is None
                and authentication is not None
                and authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
            ):
                raise ValueError(
                    "personal MCP connection requires authenticated caller"
                )
            principal_id = self._resolve_caller_principal(caller_principal_id)
            return await self._inspect_mcp_endpoint(
                endpoint=endpoint,
                authentication=authentication or MCPAuthentication.no_auth(),
                principal_id=principal_id,
            )

    async def attach_mcp_server(
        self,
        *,
        endpoint: str,
        selections: tuple[MCPToolSelection, ...],
        authentication: MCPAuthentication | None = None,
        maximum_outbound_sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL,
        local_label: str | None = None,
        binding_id: str | None = None,
        caller_principal_id: str | None = None,
    ) -> MCPBindingStatus:
        """Inspect, admit and activate exact tools before the next run."""

        if not isinstance(maximum_outbound_sensitivity, ModelSensitivity):
            raise TypeError("maximum_outbound_sensitivity is invalid")
        resolved_authentication = authentication or MCPAuthentication.no_auth()
        if (
            self._hosted
            and caller_principal_id is None
            and resolved_authentication.mode
            is MCPAuthenticationMode.PERSONAL_CONNECTION
        ):
            raise ValueError("personal MCP connection requires authenticated caller")
        principal_id = self._resolve_caller_principal(caller_principal_id)
        resolved_binding_id = binding_id or self._id_factory("mcp-binding")
        async with self._admit_owned_system_work(
            "mcp-attach", self._mcp_management_lock
        ):
            async with self._mcp_commit_lock:
                self._require_open()
                current = await self._store.load_mcp_binding(
                    self.identity.id,
                    resolved_binding_id,
                )
                self._require_mcp_management_owner(
                    current, principal_id, caller_principal_id
                )
            inspection = await self._inspect_mcp_endpoint(
                endpoint=endpoint,
                authentication=resolved_authentication,
                principal_id=principal_id,
            )
            binding = mcp_binding_from_inspection(
                binding_id=resolved_binding_id,
                agent_id=self.identity.id,
                authentication=resolved_authentication,
                maximum_outbound_sensitivity=maximum_outbound_sensitivity,
                selections=tuple(selections),
                inspection=inspection,
                local_label=local_label,
                prior=current,
                owner_principal_id=principal_id,
            )
            return await self._publish_mcp_binding(
                binding, expected_revision=None if current is None else current.revision
            )

    async def update_mcp_discovery(
        self,
        binding_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...] = (),
        caller_principal_id: str | None = None,
    ) -> MCPServerBinding:
        async with self._admit_owned_system_work(
            "mcp-discovery", self._mcp_management_lock
        ):
            async with self._mcp_commit_lock:
                self._require_open()
                principal_id = self._resolve_caller_principal(caller_principal_id)
                current = await self._store.load_mcp_binding(
                    self.identity.id, binding_id
                )
                self._require_mcp_management_owner(
                    current, principal_id, caller_principal_id
                )
                if current is None:
                    raise ValueError("MCP binding does not exist")
                updated = replace(
                    current, summary=summary, when_to_use=when_to_use, keywords=keywords
                )
            return (
                await self._publish_mcp_binding(
                    updated, expected_revision=current.revision, discovery_only=True
                )
            ).binding

    async def update_source_discovery(
        self,
        source_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...] = (),
    ) -> SourceRegistration:
        async with self._admit_owned_system_work(
            "source-discovery", self._source_management_lock
        ):
            async with self._source_commit_lock:
                self._require_open()
                return await self._store.update_source_discovery(
                    self.identity.id,
                    source_id,
                    summary=summary,
                    when_to_use=when_to_use,
                    keywords=keywords,
                )

    async def list_mcp_servers(
        self, *, caller_principal_id: str | None = None
    ) -> tuple[MCPBindingStatus, ...]:
        """Return bounded non-secret status for all independently keyed bindings."""

        self._require_open()
        principal_id = self._resolve_caller_principal(caller_principal_id)
        bindings = await self._store.list_mcp_bindings(self.identity.id)
        statuses: list[MCPBindingStatus] = []
        for binding in bindings:
            if not self._mcp_management_visible(
                binding, principal_id, caller_principal_id
            ):
                continue
            activated = self._mcp_activated_bindings.get(binding.binding_id)
            statuses.append(
                MCPBindingStatus(
                    binding,
                    None if activated is None else activated.binding.revision,
                )
            )
        return tuple(statuses)

    async def refresh_mcp_server(
        self, binding_id: str, *, caller_principal_id: str | None = None
    ) -> MCPBindingStatus:
        """Check exact remote drift and publish the checked catalog between runs."""

        async with self._admit_owned_system_work(
            "mcp-refresh", self._mcp_management_lock
        ):
            async with self._mcp_commit_lock:
                self._require_open()
                current = await self._store.load_mcp_binding(
                    self.identity.id,
                    binding_id,
                )
                if current is None:
                    raise ValueError("MCP binding does not exist")
                principal_id = self._resolve_caller_principal(caller_principal_id)
                self._require_mcp_management_owner(
                    current, principal_id, caller_principal_id
                )
                if current.state is MCPBindingState.REVOKED:
                    raise ValueError(
                        "revoked MCP binding must be explicitly reattached"
                    )
            inspection = await self._inspect_mcp_endpoint(
                endpoint=current.endpoint,
                authentication=current.authentication,
                principal_id=principal_id,
            )
            refreshed = current.checked(
                observed_at=inspection.observed_at,
                stale_reason=mcp_binding_drift_reason(current, inspection),
            )
            return await self._publish_mcp_binding(
                refreshed, expected_revision=current.revision
            )

    async def revoke_mcp_server(
        self, binding_id: str, *, caller_principal_id: str | None = None
    ) -> MCPBindingStatus:
        """Make one binding immediately unavailable without affecting siblings."""

        async with self._mcp_commit_lock:
            self._require_open()
            principal_id = self._resolve_caller_principal(caller_principal_id)
            activated = self._mcp_activated_bindings.get(binding_id)
            if activated is None:
                current = await self._store.load_mcp_binding(
                    self.identity.id,
                    binding_id,
                )
                if current is None:
                    raise ValueError("MCP binding does not exist")
                self._require_mcp_management_owner(
                    current, principal_id, caller_principal_id
                )
                if current.state is MCPBindingState.REVOKED:
                    return MCPBindingStatus(current, None)
                stored = await self._store.store_mcp_binding(
                    current.revoke(revoked_at=self._clock()),
                    expected_revision=current.revision,
                )
                return MCPBindingStatus(stored, None)

            async with activated.lock:
                current = await self._store.load_mcp_binding(
                    self.identity.id,
                    binding_id,
                )
                if current is None:
                    raise ValueError("MCP binding does not exist")
                self._require_mcp_management_owner(
                    current, principal_id, caller_principal_id
                )
                stored = (
                    current
                    if current.state is MCPBindingState.REVOKED
                    else await self._store.store_mcp_binding(
                        current.revoke(revoked_at=self._clock()),
                        expected_revision=current.revision,
                    )
                )
                client = activated.executor.client
                if client is not None:
                    await client.close()
            self._mcp_activated_bindings.pop(binding_id, None)
            return MCPBindingStatus(stored, None)

    async def _inspect_mcp_endpoint(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication,
        principal_id: str,
    ) -> MCPServerInspection:
        await check_personal_connection(
            authentication, self._mcp_connection_provider, principal_id
        )
        client = self._mcp_client_factory.create(
            endpoint=endpoint,
            authentication=authentication,
            secrets=self._secret_provider,
        )
        try:
            if authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION:
                if (
                    not isinstance(client, MCPPersonalConnectionClient)
                    or self._mcp_connection_provider is None
                ):
                    raise MCPAuthenticationError(
                        "account_unavailable",
                        "The MCP client cannot use a personal connection.",
                    )
                client.bind_personal_connection(
                    self._mcp_connection_provider, principal_id
                )
            inspection = await client.inspect(observed_at=self._clock())
            model_ids = (
                tuple(
                    candidate.provider_id for candidate in self.model_route.candidates
                )
                if self.model_route is not None
                else (() if self.model_profile is None else (self.model_profile.id,))
            )
            return replace(
                inspection,
                tools=tuple(
                    (
                        replace(tool, supported=False, unsupported_reason=reason)
                        if tool.supported
                        and tool.input_schema is not None
                        and (
                            reason := tool_schema_incompatibility(
                                tool.input_schema, model_ids
                            )
                        )
                        is not None
                        else tool
                    )
                    for tool in inspection.tools
                ),
            )
        finally:
            await client.close()

    def _mcp_management_visible(
        self,
        binding: MCPServerBinding,
        principal_id: str,
        caller_principal_id: str | None,
    ) -> bool:
        return binding.owner_principal_id == principal_id and not (
            self._hosted
            and caller_principal_id is None
            and binding.authentication.mode is MCPAuthenticationMode.PERSONAL_CONNECTION
        )

    def _require_mcp_management_owner(
        self,
        binding: MCPServerBinding | None,
        principal_id: str,
        caller_principal_id: str | None,
    ) -> None:
        if binding is not None and not self._mcp_management_visible(
            binding, principal_id, caller_principal_id
        ):
            raise ValueError("MCP binding does not exist")

    async def _publish_mcp_binding(
        self,
        binding: MCPServerBinding,
        *,
        expected_revision: int | None,
        discovery_only: bool = False,
    ) -> MCPBindingStatus:
        """Stage first, CAS persistence, then switch every catalog reader together."""
        async with self._mcp_commit_lock:
            self._require_open()
            bindings = await self._store.list_mcp_bindings(self.identity.id)
            next_bindings = tuple(
                item for item in bindings if item.binding_id != binding.binding_id
            ) + (binding,)
            previous = tuple(self._mcp_activated_bindings.values())
            activated, publish = await self._stage_mcp_catalog(next_bindings, previous)
            retired = [item for item in previous if item not in activated]
            if discovery_only:
                current = next(
                    (
                        item
                        for item in bindings
                        if item.binding_id == binding.binding_id
                    ),
                    None,
                )
                if current is None or current.revision != expected_revision:
                    raise ValueError("MCP binding revision precondition failed")
                stored = await self._store.update_mcp_discovery(
                    self.identity.id,
                    binding.binding_id,
                    summary=binding.summary,
                    when_to_use=binding.when_to_use,
                    keywords=binding.keywords,
                )
            else:
                stored = await self._store.store_mcp_binding(
                    binding, expected_revision=expected_revision
                )
            self._retired_mcp_bindings.extend(retired)
            publish()
        for item in retired:
            await item.executor.close()
            self._retired_mcp_bindings.remove(item)
        return MCPBindingStatus(
            stored,
            stored.revision if stored.state is MCPBindingState.ACTIVE else None,
        )

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._attach_source(source, attached_at=self._clock())

    async def _attach_source(
        self,
        source: ResourceSource,
        *,
        attached_at: datetime,
    ) -> SourceRegistration:
        async with self._admit_owned_system_work(
            "source-attach", self._source_management_lock
        ):
            async with self._source_commit_lock:
                self._require_open()
                return await self._attach_source_locked(
                    source,
                    attached_at=attached_at,
                )

    async def _attach_source_locked(
        self,
        source: ResourceSource,
        *,
        attached_at: datetime,
        persisted_registration: SourceRegistration | None = None,
    ) -> SourceRegistration:
        adapter = await source.open(
            agent_id=self.identity.id,
            attached_at=attached_at,
            clock=self._clock,
        )
        if not isinstance(adapter, ResourceAdapter):
            raise TypeError("source open() must return ResourceAdapter")
        opened_registration = adapter.registration
        registration = opened_registration
        sync: CatalogSync | None = None
        try:
            self._source_types.require(registration.adapter_id)
            if persisted_registration is not None:
                current_identity = replace(
                    opened_registration,
                    summary=persisted_registration.summary,
                    when_to_use=persisted_registration.when_to_use,
                    keywords=persisted_registration.keywords,
                    presentation_sensitivity=persisted_registration.presentation_sensitivity,
                )
                if persisted_registration != current_identity:
                    raise ValueError(
                        "refreshed source registration disagrees with persisted identity"
                    )
                registration = persisted_registration
            sync = CatalogSync(
                id=self._id_factory("catalog-sync"),
                agent_id=self.identity.id,
                source_id=registration.id,
                adapter_id=registration.adapter_id,
                status=CatalogSyncStatus.RUNNING,
                started_at=self._clock(),
            )
            await self._store.record_sync(sync)
            discovery = await adapter.discover(
                DiscoveryRequest(
                    agent_id=self.identity.id,
                    source_id=registration.id,
                    sync_id=sync.id,
                    requested_at=sync.started_at,
                    agent_root=None if self._home is None else self._home.parent.parent,
                    agent_home=self._home,
                )
            )
            await self._store.commit_snapshot(
                discovery.snapshot,
                registration=registration,
            )
            self._admission_coordinator.configure_source_resource_capacity(
                f"source:{registration.id}",
                self._source_types.read_capacity(
                    registration.adapter_id, self._host_capacity
                ),
            )
            return registration
        except BaseException as error:
            if isinstance(error, ResourceAdapterError):
                code = error.code
            else:
                code = "source_attach_failed"
            if sync is not None:
                try:
                    await self._store.record_sync(
                        CatalogSync(
                            id=sync.id,
                            agent_id=self.identity.id,
                            source_id=registration.id,
                            adapter_id=registration.adapter_id,
                            status=CatalogSyncStatus.FAILED,
                            started_at=sync.started_at,
                            completed_at=max(self._clock(), sync.started_at),
                            error_code=code,
                        )
                    )
                except BaseException:
                    pass
            raise
        finally:
            await adapter.close()

    async def edit_source(
        self,
        source_id: str,
        source: ResourceSource,
        *,
        confirmation_handler: SourceEditConfirmationHandler,
    ) -> SourceEditResult | None:
        """Validate, review, and atomically edit one active source connection."""

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        if not isinstance(source, ResourceSource):
            raise TypeError("source must implement ResourceSource")
        if not callable(confirmation_handler):
            raise TypeError("confirmation_handler must be callable")
        async with self._admit_owned_system_work(
            "source-edit", self._source_management_lock
        ):
            async with self._source_commit_lock:
                self._require_open()
                current = await self._store.load_source(self.identity.id, source_id)
                if current is None or not current.active:
                    raise ValueError("source edit requires an active owned source")
                self._source_types.require(current.adapter_id)
                current_read_scope = await self._store.load_source_read_scope(
                    self.identity.id,
                    source_id,
                )
                if current_read_scope is None:
                    raise ValueError("active source is missing its read scope")
                current_resources = await self._store.list_resources(
                    self.identity.id,
                    source_id,
                )
                attached_at = self._clock()
                adapter = await source.open(
                    agent_id=self.identity.id,
                    attached_at=attached_at,
                    clock=self._clock,
                )
                if not isinstance(adapter, ResourceAdapter):
                    raise TypeError("source open() must return ResourceAdapter")
                committed = False
                try:
                    registration = adapter.registration
                    if registration.adapter_id != current.adapter_id:
                        raise ValueError("source edit cannot change source type")
                    if registration.id == current.id:
                        registration = replace(
                            registration,
                            summary=current.summary,
                            when_to_use=current.when_to_use,
                            keywords=current.keywords,
                            presentation_sensitivity=current.presentation_sensitivity,
                        )
                    if registration.id != current.id:
                        existing = await self._store.load_source(
                            self.identity.id,
                            registration.id,
                        )
                        if existing is not None and existing.active:
                            raise ValueError(
                                "replacement connection is already attached"
                            )
                    sync = CatalogSync(
                        id=self._id_factory("catalog-sync"),
                        agent_id=self.identity.id,
                        source_id=registration.id,
                        adapter_id=registration.adapter_id,
                        status=CatalogSyncStatus.RUNNING,
                        started_at=self._clock(),
                    )
                    discovery = await adapter.discover(
                        DiscoveryRequest(
                            agent_id=self.identity.id,
                            source_id=registration.id,
                            sync_id=sync.id,
                            requested_at=sync.started_at,
                        )
                    )
                    read_scope, omitted = _source_edit_read_scope(
                        current_read_scope=current_read_scope,
                        current_resources=current_resources,
                        replacement=discovery.snapshot,
                    )
                    preview = SourceEditPreview(
                        current_source_id=current.id,
                        replacement_source_id=registration.id,
                        display_name=registration.display_name,
                        adapter_id=registration.adapter_id,
                        resource_count=len(discovery.snapshot.resources),
                        relationship_count=len(discovery.snapshot.relationships),
                        read_mode=read_scope.mode,
                        preserved_read_resource_count=(
                            len(discovery.snapshot.resources)
                            if read_scope.mode is SourceReadMode.ALL
                            else len(read_scope.resource_ids)
                        ),
                        omitted_read_resources=omitted,
                    )
                    confirmed = await confirmation_handler(preview)
                    if not isinstance(confirmed, bool):
                        raise TypeError("source edit confirmation must return bool")
                    if not confirmed:
                        return None
                    await self._store.commit_source_edit(
                        discovery.snapshot,
                        registration=registration,
                        replaced_source_id=current.id,
                        replaced_at=max(self._clock(), current.attached_at),
                        read_scope=read_scope,
                    )
                    committed = True
                    new_references = _owned_source_credential_references(
                        registration.adapter_id,
                        registration.configuration,
                        agent_id=self.identity.id,
                        source_types=self._source_types,
                    )
                    previous_credential_deleted = True
                    for old_reference in _owned_source_credential_references(
                        current.adapter_id,
                        current.configuration,
                        agent_id=self.identity.id,
                        source_types=self._source_types,
                    ):
                        if old_reference in new_references:
                            continue
                        try:
                            await asyncio.wait_for(
                                self._keychain.delete(old_reference),
                                timeout=_CREDENTIAL_CLEANUP_TIMEOUT_SECONDS,
                            )
                        except BaseException:
                            previous_credential_deleted = False
                    return SourceEditResult(
                        source=registration,
                        identity_changed=registration.id != current.id,
                        previous_credential_deleted=previous_credential_deleted,
                    )
                finally:
                    try:
                        await adapter.close()
                    except BaseException:
                        if not committed:
                            raise

    async def attach_sqlite(
        self,
        path: str | Path,
        *,
        name: str | None = None,
    ) -> SourceRegistration:
        """Attach one ordinary read-only SQLite source."""

        return await self.attach(SQLiteSource(path=path, name=name))

    async def edit_sqlite_source(
        self,
        source_id: str,
        path: str | Path,
        *,
        confirmation_handler: SourceEditConfirmationHandler,
        name: str | None = None,
    ) -> SourceEditResult | None:
        return await self.edit_source(
            source_id,
            SQLiteSource(path=path, name=name),
            confirmation_handler=confirmation_handler,
        )

    async def store_postgresql_password(self, password: str) -> SecretReference:
        """Store one database password for an in-process onboarding attempt."""

        if (
            not isinstance(password, str)
            or not password
            or len(password.encode("utf-8")) > 64 * 1_024
        ):
            raise ValueError("PostgreSQL password must be non-empty and at most 64 KiB")
        async with self._credential_management_lock:
            self._require_open()
            reference = SecretReference.keychain(
                _credential_account(
                    self.identity.id,
                    POSTGRESQL_ADAPTER_ID,
                    self._id_factory("credential"),
                )
            )
            credential = password
            password = ""
            try:
                await self._keychain.set(reference, credential)
                return reference
            except BaseException:
                try:
                    await self._keychain.delete(reference)
                except BaseException:
                    pass
                raise
            finally:
                credential = ""

    async def delete_postgresql_password(
        self,
        reference: SecretReference,
    ) -> None:
        """Delete one exact credential created for this agent's source setup."""

        if (
            not isinstance(reference, SecretReference)
            or reference.scheme != "keychain"
            or not reference.name.startswith(
                _credential_account_prefix(self.identity.id, POSTGRESQL_ADAPTER_ID)
            )
        ):
            raise ValueError(
                "credential does not belong to this agent's PostgreSQL setup"
            )
        async with self._credential_management_lock:
            self._require_open()
            await self._keychain.delete(reference)

    async def probe_postgresql(
        self,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        port: int = 5432,
        ssl_mode: str = "require",
        public_network_only: bool = False,
    ) -> PostgreSQLProbeResult:
        """Run the adapter-owned read-only schema probe without persistence."""

        self._require_open()
        source = PostgreSQLSource(
            host=host,
            port=port,
            database=database,
            username=username,
            credential=credential,
            schemas=("public",),
            ssl_mode=ssl_mode,
            public_network_only=public_network_only,
            secret_provider=self._secret_provider or self._keychain,
        )
        return await source.probe()

    async def attach_postgresql(
        self,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        schemas: tuple[str, ...],
        port: int = 5432,
        ssl_mode: str = "require",
        public_network_only: bool = False,
        name: str | None = None,
    ) -> SourceRegistration:
        """Construct and attach one ordinary selected-schema PostgreSQL source."""

        return await self.attach(
            PostgreSQLSource(
                host=host,
                port=port,
                database=database,
                username=username,
                credential=credential,
                schemas=schemas,
                ssl_mode=ssl_mode,
                public_network_only=public_network_only,
                name=name,
                secret_provider=self._secret_provider or self._keychain,
            )
        )

    async def edit_postgresql_source(
        self,
        source_id: str,
        *,
        host: str,
        database: str,
        username: str,
        credential: SecretReference,
        schemas: tuple[str, ...],
        confirmation_handler: SourceEditConfirmationHandler,
        port: int = 5432,
        ssl_mode: str = "require",
        public_network_only: bool = False,
        name: str | None = None,
    ) -> SourceEditResult | None:
        """Edit PostgreSQL through the ordinary validated source boundary."""

        return await self.edit_source(
            source_id,
            PostgreSQLSource(
                host=host,
                port=port,
                database=database,
                username=username,
                credential=credential,
                schemas=schemas,
                ssl_mode=ssl_mode,
                public_network_only=public_network_only,
                name=name,
                secret_provider=self._secret_provider or self._keychain,
            ),
            confirmation_handler=confirmation_handler,
        )

    async def inspect_source_permissions(
        self,
        source_id: str,
    ) -> SourcePermissionsInspection:
        """Inspect exact scopes against complete trusted current catalog truth."""

        async with self._source_commit_lock:
            self._require_open()
            return await self._inspect_source_permissions_locked(source_id)

    async def preview_source_permissions(
        self,
        *,
        source_id: str,
        read_mode: SourceReadMode,
        read_resource_ids: tuple[str, ...],
        relational_write_scopes: Mapping[str, Mapping[str, object]],
    ) -> SourcePermissionsPreview:
        """Build and retain one bounded in-process exact confirmation preview."""

        async with self._admit_owned_system_work(
            "source-permission-preview", self._source_permission_lock
        ):
            async with self._source_commit_lock:
                self._require_open()
                inspection = await self._inspect_source_permissions_locked(source_id)
                preview = await self._build_source_permissions_preview(
                    inspection,
                    read_mode=read_mode,
                    read_resource_ids=read_resource_ids,
                    relational_write_scopes=relational_write_scopes,
                )
                self._source_permission_previews[source_id] = preview
                return preview

    async def apply_source_permissions(
        self,
        *,
        source_id: str,
        confirmation_fingerprint: str,
    ) -> SourcePermissionsInspection:
        """Revalidate and atomically apply one exact in-process preview."""

        async with self._admit_owned_system_work(
            "source-permission-apply", self._source_permission_lock
        ):
            async with self._source_commit_lock:
                self._require_open()
                preview = self._source_permission_previews.get(source_id)
                if preview is None or not hmac.compare_digest(
                    preview.confirmation_fingerprint,
                    confirmation_fingerprint,
                ):
                    raise ValueError(
                        "permission confirmation is unknown or expired; preview again"
                    )
                current = await self._inspect_source_permissions_locked(source_id)
                if current.catalog_generation != preview.catalog_generation:
                    raise ValueError(
                        "source catalog changed after preview; preview again"
                    )
                if current.state == preview.after:
                    return current
                if current.state != preview.before:
                    raise ValueError(
                        "source permissions changed after preview; preview again"
                    )

                update_mapping = {
                    scope.resource_id: {
                        key: value
                        for key, value in scope.constraints().items()
                        if key != "resource_revision"
                    }
                    for scope in preview.after.relational_write_scopes
                }
                revalidated = await self._build_source_permissions_preview(
                    current,
                    read_mode=preview.after.read_scope.mode,
                    read_resource_ids=preview.after.read_scope.resource_ids,
                    relational_write_scopes=update_mapping,
                )
                if revalidated.after != preview.after:
                    raise ValueError(
                        "source permission facts changed after preview; preview again"
                    )
                await self._store.replace_source_permission_scopes(
                    preview.after.read_scope,
                    preview.after.relational_write_scopes,
                )
                inspection = await self._inspect_source_permissions_locked(source_id)
            await self._retire_source_backends(source_id)
            return inspection

    async def _inspect_source_permissions_locked(
        self,
        source_id: str,
    ) -> SourcePermissionsInspection:
        return await _inspect_source_permissions(
            self._store,
            self._data_view,
            self._source_types,
            self.identity.id,
            source_id,
        )

    async def _build_source_permissions_preview(
        self,
        inspection: SourcePermissionsInspection,
        *,
        read_mode: SourceReadMode,
        read_resource_ids: tuple[str, ...],
        relational_write_scopes: Mapping[str, Mapping[str, object]],
    ) -> SourcePermissionsPreview:
        return await _build_source_permissions_preview(
            self._store,
            self._catalog_service,
            self._data_view,
            self._source_types,
            self.identity.id,
            self._source_permission_confirmation_key,
            inspection,
            read_mode=read_mode,
            read_resource_ids=read_resource_ids,
            relational_write_scopes=relational_write_scopes,
        )

    async def relational_upsert_readiness(
        self, source_id: str, resource_id: str
    ) -> FrozenJsonObject:
        async with self._source_commit_lock:
            self._require_open()
            permission = await self._data_view.load_relational_write_scope(
                self.identity.id, source_id, resource_id
            )
            if permission is None:
                raise ValueError("exact current upsert permission is required")
            constraints = FrozenJsonObject.from_mapping(
                {
                    "source_id": source_id,
                    "resource_id": resource_id,
                    **permission.constraints(),
                }
            )
            return await _relational_upsert_readiness(
                self._data_view, self._source_backends, self.identity.id, constraints
            )

    async def relational_update_readiness(
        self,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
    ) -> RelationalUpdateReadiness:
        """Inspect one exact PostgreSQL update scope without mutating state."""

        self._require_open()
        # The PostgreSQL readiness record also reports other sources as unavailable.
        backend = self._source_backends[SourceOperation.RELATIONAL_UPDATE][
            POSTGRESQL_ADAPTER_ID
        ]
        return await backend.relational_update_readiness(
            agent_id=self.identity.id,
            source_id=source_id,
            resource_id=resource_id,
            assignment_columns=assignment_columns,
        )

    async def _retire_source_backends(self, source_id: str) -> None:
        for backend in _backends_with(self._source_backends, "retire_source"):
            await backend.retire_source(self.identity.id, source_id)

    async def detach(self, source_id: str) -> SourceRegistration:
        async with self._admit_owned_system_work(
            "source-detach", self._source_management_lock
        ):
            async with self._source_commit_lock:
                self._require_open()
                detached = await self._store.detach_source(
                    self.identity.id, source_id, self._clock()
                )
                references = _owned_source_credential_references(
                    detached.adapter_id,
                    detached.configuration,
                    agent_id=self.identity.id,
                    source_types=self._source_types,
                )
            await self._retire_source_backends(source_id)
            for reference in references:
                try:
                    await self._keychain.delete(reference)
                except Exception:
                    raise AgentHomeError(
                        "source was detached, but its stored credential could not be deleted"
                    ) from None
            return detached

    async def refresh_source(self, source_id: str) -> SourceRegistration:
        """Refresh one active source using its exact admitted registration."""

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        async with self._admit_owned_system_work(
            "source-refresh", self._source_management_lock
        ):
            async with self._source_commit_lock:
                self._require_open()
                registration = await self._store.load_source(
                    self.identity.id, source_id
                )
                if registration is None or not registration.active:
                    raise ValueError("unknown active source for this agent")
                try:
                    source = self._source_types.require(
                        registration.adapter_id
                    ).from_registration(
                        registration,
                        secret_provider=default_secret_provider(
                            self._secret_provider or self._keychain
                        ),
                    )
                except (TypeError, ValueError) as error:
                    raise AgentHomeError(str(error)) from error
                return await self._attach_source_locked(
                    source,
                    attached_at=registration.attached_at,
                    persisted_registration=registration,
                )

    async def list_sources(self) -> tuple[SourceRegistration, ...]:
        self._require_open()
        return await self._store.list_sources(self.identity.id)

    async def unavailable_sources(self) -> tuple[SourceRegistration, ...]:
        """Active sources whose type is not registered in this composition."""

        return tuple(
            registration
            for registration in await self.list_sources()
            if registration.active
            and self._source_types.get(registration.adapter_id) is None
        )

    async def list_catalog_resources(
        self, *, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        self._require_open()
        return await self._store.list_resources(self.identity.id, source_id)

    async def catalog_summary(self) -> CatalogSummary:
        """Return current committed catalog facts as one consistent projection."""

        async with self._source_commit_lock:
            self._require_open()
            return await self._catalog_service.summary(self.identity.id)

    async def catalog_preview(
        self,
        *,
        limit: int = 12,
    ) -> tuple[CatalogResource, ...]:
        """Return a bounded deterministic preview from current catalog truth."""

        async with self._source_commit_lock:
            self._require_open()
            return await self._catalog_service.preview(
                self.identity.id,
                limit=limit,
            )

    async def search_catalog(
        self, request: CatalogSearchRequest
    ) -> CatalogSearchResult:
        self._require_open()
        if request.agent_id != self.identity.id:
            raise ValueError("catalog search belongs to another agent")
        return await self._catalog_service.search(request)

    async def inspect_catalog_resource(self, resource_id: str) -> FrozenJsonObject:
        self._require_open()
        return await self._catalog_service.inspect_resource(
            self.identity.id, resource_id
        )

    async def declare_catalog_link(
        self,
        from_resource_id: str,
        to_resource_id: str,
        field_pairs: tuple[tuple[str, str], ...],
        *,
        note: str | None = None,
    ) -> CatalogLink:
        pairs = tuple(
            RelationshipFieldPair(source_field, target_field, ordinal)
            for ordinal, (source_field, target_field) in enumerate(field_pairs)
        )
        async with self._source_commit_lock:
            self._require_open()
            return await self._catalog_service.declare_link(
                self.identity.id,
                from_resource_id,
                to_resource_id,
                pairs,
                declared_at=self._clock(),
                note=note,
            )

    async def remove_catalog_link(self, link_id: str) -> bool:
        async with self._source_commit_lock:
            self._require_open()
            return await self._catalog_service.remove_link(self.identity.id, link_id)

    async def list_catalog_links(self) -> tuple[CatalogLinkStatus, ...]:
        async with self._source_commit_lock:
            self._require_open()
            return await self._catalog_service.link_statuses(self.identity.id)

    @asynccontextmanager
    async def _admit_system_work(self, label: str):
        deadline = (
            asyncio.get_running_loop().time() + self._limits.max_wall_time_seconds
        )
        try:
            lease = await self._admission_coordinator.admit_execution(
                WorkloadClass.SYSTEM,
                self._id_factory(label),
                absolute_deadline=deadline,
            )
        except AdmissionClosedError as error:
            raise AgentHomeError("embedded agent is closed") from error
        async with lease:
            yield lease

    async def _provider_admission(self, key: str, deadline: float | None):
        return await self._admission_coordinator.provider_permit(
            key,
            deadline=deadline,
        )

    @asynccontextmanager
    async def _admit_owned_system_work(
        self,
        label: str,
        owner_lock: asyncio.Lock,
    ):
        async with self._admit_system_work(label) as lease:
            async with owner_lock:
                yield lease

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
        close_task = self._close_task
        cancelled = False
        while not close_task.done():
            try:
                await asyncio.shield(close_task)
            except asyncio.CancelledError:
                cancelled = True
        close_task.result()
        if cancelled:
            raise asyncio.CancelledError

    async def _finish_close(self) -> None:
        first_error: BaseException | None = None
        await self._admission_coordinator.begin_draining()
        try:
            await self._routine_supervisor.close()
        except BaseException as error:
            first_error = error
        try:
            await self._job_supervisor.close()
        except BaseException as error:
            first_error = error
        drain_deadline = (
            asyncio.get_running_loop().time() + _RUN_ADMISSION_DRAIN_SECONDS
        )
        await self._capability_runtime.handoff_analysis_recovery()
        try:
            await self._admission_coordinator.close(deadline=drain_deadline)
        except BaseException as error:
            # Shared dependencies and the process writer lease must stay alive if
            # an admitted owner could not be cancelled and settled by the barrier.
            # A failed close is deliberately fail-closed rather than tearing the
            # provider/store out from underneath live work.
            if first_error is not None:
                raise error from first_error
            raise
        for backend in (
            *_backends_with(self._source_backends, "close"),
            *(
                ()
                if self._combined_query_engine is None
                else (self._combined_query_engine,)
            ),
        ):
            try:
                await backend.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        owned_model_provider = self._owned_model_provider
        model_shutdown_deadline = (
            asyncio.get_running_loop().time()
            + self._model_call_policy.cleanup_timeout_seconds
        )
        if owned_model_provider is not None:
            try:
                await owned_model_provider.close(deadline=model_shutdown_deadline)
                self._owned_model_provider = None
            except BaseException as error:
                if first_error is None:
                    first_error = error
        activated_bindings = (
            *self._mcp_activated_bindings.values(),
            *self._retired_mcp_bindings,
        )
        self._mcp_activated_bindings.clear()
        self._retired_mcp_bindings.clear()
        for activated in activated_bindings:
            try:
                await activated.executor.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if self._workspace_backend is not None:
            try:
                await self._workspace_backend.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        for store in (
            self._memory_store,
            self._skill_store,
            *(
                (self._artifact_delivery,)
                if self._artifact_delivery is not None
                else ()
            ),
            self._artifact_store,
            *((self._store,) if self._close_state else ()),
        ):
            try:
                await store.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if self._owns_credential_session:
            try:
                await self._keychain.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if self._writer_lock is not None:
            self._writer_lock.release()
        if first_error is not None:
            raise first_error

    def _require_loop(self) -> AgentLoop:
        self._require_open()
        if self._loop is None:
            raise AgentNotConfiguredError("agent execution requires a model")
        return self._loop

    def _require_artifact_delivery(self) -> LocalArtifactDelivery:
        if self._artifact_delivery is None:
            raise AgentHomeError("hosted composition has no local artifact delivery")
        return self._artifact_delivery

    def _require_open(self) -> None:
        if self._closed:
            raise AgentHomeError("embedded agent is closed")


def _require_value(value: _T | None) -> _T:
    if value is None:
        raise AgentNotConfiguredError("agent loop is incomplete")
    return value


async def _await_sync_completion(callback: Callable[[], _T]) -> tuple[_T, bool]:
    worker = asyncio.create_task(asyncio.to_thread(callback))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
            continue
    return worker.result(), cancelled


async def _recover_analysis_evidence(store: StateStore, agent_id: str) -> None:
    """Settle analysis evidence that a crashed host left admitted or running."""
    from ..adapters.analytical_workspace.recovery import (
        dispose_generation_proof,
        recover_generation,
    )
    from ..loop.analysis import AnalysisEvidence

    after = ("", "")
    while page := await store.pending_analysis_evidence(agent_id, after=after):
        for pending in page:
            facts = pending.facts
            if pending.kind == "generation":
                if facts.get("proof_disposed") is True:
                    continue
                recovered = facts.get("recovery")
                if recovered is None and facts["status"] in {
                    "admitted",
                    "running",
                    "cleanup_failed",
                }:
                    recovery = await recover_generation(facts)
                    pending = await store.recover_analysis_evidence(
                        agent_id, pending, recovery
                    )
                elif recovered is None and facts["status"] != "closed":
                    continue
                # Closure facts are durable before proof disposal. A
                # crash on either side of unlink can repeat this step.
                dispose_generation_proof(pending.facts)
                await store.recover_analysis_evidence(
                    agent_id, pending, proof_disposed=True
                )
            elif (
                facts["status"] in {"admitted", "running"}
                and await store.result(pending.run_id) is None
            ):
                await store.record_analysis_evidence(
                    agent_id,
                    AnalysisEvidence(
                        pending.run_id,
                        pending.evidence_id,
                        pending.kind,
                        {
                            **facts,
                            "status": "interrupted",
                            "interruption": "host_crash",
                            "completion_observed": False,
                        },
                    ),
                )
        last = page[-1]
        after = (last.run_id, last.evidence_id)


async def _recover_agent_state(
    store: StateStore, agent_id: str, clock: Callable[[], datetime]
) -> None:
    _, cancelled = await _await_async_completion(
        lambda: store.recover_started_effect_receipts(agent_id, recovered_at=clock())
    )
    if cancelled:
        raise asyncio.CancelledError
    _, cancelled = await _await_async_completion(
        lambda: store.recover_unfinished_runs(agent_id, created_at=clock())
    )
    if cancelled:
        raise asyncio.CancelledError


async def _await_async_completion(
    callback: Callable[[], Awaitable[_T]],
) -> tuple[_T, bool]:
    worker = asyncio.ensure_future(callback())
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
            continue
    return worker.result(), cancelled


__all__ = [
    "AgentAlreadyExistsError",
    "AgentHomeError",
    "AgentIdentityMismatchError",
    "AgentModelConfigurationError",
    "AgentNameError",
    "AgentNotConfiguredError",
    "AgentNotFoundError",
    "EmbeddedAgent",
    "HostActiveError",
]
