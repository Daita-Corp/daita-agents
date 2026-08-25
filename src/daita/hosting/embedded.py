"""Compose and own the in-process runtime for one persistent agent home."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import stat
import tomllib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Self, TypeVar, cast
from uuid import uuid4

from .._json import FrozenJsonObject, canonical_json
from ..adapters.local_files import LocalDirectoryReadBackend, LocalDirectorySource
from ..adapters.job_profiles import ConnectedJobProfile
from ..adapters.mcp import (
    MCPAuthentication,
    MCPBindingState,
    MCPBindingStatus,
    MCPClientFactory,
    MCPServerInspection,
    MCPToolSelection,
    StreamableHTTPMCPClientFactory,
    mcp_binding_drift_reason,
    mcp_binding_from_inspection,
)
from ..adapters.models import DiscoveryRequest, SourceRegistration
from ..adapters.postgresql import PostgreSQLProbeResult, PostgreSQLSource
from ..adapters.postgresql_query import PostgreSQLQueryBackend
from ..adapters.postgresql_write import (
    PostgreSQLUpdatePreviewBackend,
    PostgreSQLUpdateReadiness,
)
from ..adapters.protocols import ResourceAdapter, ResourceAdapterError, ResourceSource
from ..adapters.sqlite import SQLiteSource
from ..adapters.sqlite_query import SQLiteQueryBackend
from ..artifacts.delivery import LocalArtifactDelivery
from ..artifacts.models import (
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactPayload,
)
from ..artifacts.store import AgentHomeArtifactStore
from ..autonomy import (
    FOLLOWUP_INSTRUCTION,
    FOLLOWUP_INSTRUCTION_DIGEST,
    FOLLOWUP_INSTRUCTION_ID,
    FOLLOWUP_LEASE_SECONDS,
    FollowupDisposition,
    InboxItem,
    create_terminal_job_followup,
)
from ..capabilities import (
    AccessMode,
    ApprovalHandler,
    CapabilityDeclarations,
    CapabilityRegistry,
    OperationalEffect,
)
from ..capability_runtime import CapabilityRuntime
from ..catalog.capabilities import (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_SCHEMA_CAPABILITY_ID,
    catalog_declarations,
)
from ..catalog.models import (
    CatalogResource,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSummary,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    SourceCatalogSnapshot,
)
from ..catalog.service import CatalogService
from ..config import AgentConfig
from ..domains.data import (
    ARTIFACT_DOMAIN_OWNER_ID,
    CatalogDataView,
    DataCapabilityDomain,
    DataContextBuilder,
    ArtifactCapabilityDomain,
    artifact_declarations,
    local_file_read_declarations,
    postgresql_query_declarations,
    postgresql_update_declarations,
    postgresql_update_preview_declarations,
    sqlite_query_declarations,
    LOCAL_FILE_READ_CAPABILITY_ID,
    POSTGRESQL_QUERY_CAPABILITY_ID,
    SQLITE_QUERY_CAPABILITY_ID,
)
from ..domains.data.controller import DATA_DOMAIN_OWNER_ID
from ..domains.learning import LearningCandidateGuard
from ..domains.mcp import MCPActivatedBinding, activate_mcp_domain
from ..domains.data.context import _project_completed_history
from ..domains.data.profile_jobs import (
    DATA_PROFILE_DOMAIN_OWNER_ID,
    DataProfileAdmission,
    DataProfileCapabilityDomain,
    data_profile_declarations,
)
from ..domains.data.sql import (
    validate_postgresql_update_scope,
)
from ..errors import AgentError, StateCompatibilityCode, StateCompatibilityError
from ..identity import AgentIdentity
from ..learning_candidates import (
    LEARNING_REVIEW_MAX_TOTAL_TOKENS,
    LearningCandidate,
    LearningCandidateContent,
    LearningCandidateRejectionReason,
    LearningCandidateStatus,
    LearningCandidateView,
    LearningReviewResult,
    OneShotCandidateReviewer,
)
from ..llm.errors import ModelProviderError, ProviderErrorCode
from ..llm.factory import create_model_route_provider
from ..llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelProfile,
    ModelRequest,
    ModelSensitivity,
    TextBlock,
    ToolDefinition,
)
from ..llm.profiles import reviewed_model_profile
from ..llm.protocols import ModelProvider
from ..llm.routing import (
    ModelProviderRegistration,
    ModelRoute,
    ModelRouteCandidate,
    ModelRouter,
    RetryPolicy,
)
from ..llm.subscription_auth import CodexDevicePrompt, login_codex_subscription
from ..loop.driver import (
    AgentLoop,
    ContextBuilder,
    LoopPreparationError,
    ToolRuntime,
)
from ..loop.models import (
    ConversationRun,
    LoopExit,
    LoopLimits,
    RunInput,
    RunOrigin,
    RunStartEnvelope,
    Transcript,
)
from ..jobs.capabilities import (
    JOB_DOMAIN_OWNER_ID,
    JOB_INSPECT_CAPABILITY_ID,
    JOB_READ_RESULTS_CAPABILITY_ID,
    JobCapabilityDomain,
    job_capability_declarations,
)
from ..jobs.models import (
    JobCompletionOwnerKind,
    JobExecutionMode,
    JobInspection,
    JobResultView,
    JobStatus,
    JobSummary,
)
from ..jobs.owner import JobOwner
from ..jobs.supervisor import JobSupervisor
from ..memory import MemoryStore
from ..memory.capabilities import (
    MEMORY_DOMAIN_OWNER_ID,
    MemoryCapabilityDomain,
    memory_set_declarations,
)
from ..observation import AgentObserver
from ..security import (
    CredentialSession,
    KeychainSecretProvider,
    KeychainStore,
    SecretProvider,
    SecretReference,
    default_secret_provider,
)
from ..semantics import (
    SemanticAnnotation,
    SemanticAnnotationState,
    SemanticAnnotationView,
    SemanticKind,
    SemanticCapabilityDomain,
    SEMANTIC_DOMAIN_OWNER_ID,
    inspect_semantic_annotations,
    semantic_declarations,
)
from ..skills import Skill, SkillStore, SkillSummary
from ..skills.capabilities import (
    SKILL_DOMAIN_OWNER_ID,
    SkillCapabilityDomain,
    skill_declarations,
)
from ..storage.sqlite import SQLiteStateStore
from ..storage.sqlite_records import (
    PostgreSQLUpdateScope,
    SourcePermissionResource,
    SourcePermissionsInspection,
    SourcePermissionsPreview,
    SourcePermissionState,
    SourcePermissionSummary,
    SourceReadMode,
    SourceReadScope,
    postgresql_update_authorization_fingerprint,
)

_AGENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_CONVERSATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_MAX_AGENT_HOME_CANDIDATES = 256
_MAX_DISCOVERED_AGENTS = 100
_LEGACY_STATE_ROOT_MARKERS = (
    "catalog.json",
    "graph",
    "memory",
    "sessions",
)
_MODEL_CONFIG_NAME = "config.json"
_MAX_MODEL_CONFIG_BYTES = 64 * 1_024
_CREDENTIAL_CLEANUP_TIMEOUT_SECONDS = 1.0
_MODEL_VALIDATION_TOOL_NAME = "daita_validate_tool_support"
_MODEL_VALIDATION_MAX_OUTPUT_TOKENS = 16
_REASONING_MODEL_VALIDATION_MAX_OUTPUT_TOKENS = 25_000
_CANDIDATE_REVIEWER_MAX_OUTPUT_TOKENS = LEARNING_REVIEW_MAX_TOTAL_TOKENS // 4
_PROVIDER_NAME = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
_SOURCE_ALIAS_SEPARATOR = re.compile(r"[^a-z0-9]+")
_SUBSCRIPTION_PROVIDERS = frozenset({"codex", "claude-code", "grok-build"})
_SUBSCRIPTION_CREDENTIAL_PROVIDERS = frozenset({"codex"})
_BUILTIN_PROVIDERS = frozenset(
    {"openai", "anthropic", "gemini", "grok", "ollama", *_SUBSCRIPTION_PROVIDERS}
)
_T = TypeVar("_T")
_STAGE_C_ALLOWED_CAPABILITY_IDS = (
    CATALOG_INSPECT_CAPABILITY_ID,
    CATALOG_SCHEMA_CAPABILITY_ID,
    LOCAL_FILE_READ_CAPABILITY_ID,
    POSTGRESQL_QUERY_CAPABILITY_ID,
    SQLITE_QUERY_CAPABILITY_ID,
    JOB_INSPECT_CAPABILITY_ID,
    JOB_READ_RESULTS_CAPABILITY_ID,
)
_STAGE_C_SAFETY_WAKE_SECONDS = 5.0


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _stage_c_capability_ids(registry: CapabilityRegistry) -> tuple[str, ...]:
    return registry.validate_execution_scope_grant(
        _STAGE_C_ALLOWED_CAPABILITY_IDS,
        allowed_access_modes=frozenset({AccessMode.NONE, AccessMode.READ}),
        allowed_operational_effects=frozenset({OperationalEffect.NONE}),
    )


def _stage_c_model_routes(
    model: ModelProvider | None,
    model_route: ModelRoute | None,
) -> tuple[str, ...]:
    if model_route is not None:
        return tuple(candidate.provider_id for candidate in model_route.candidates)
    if isinstance(model, ModelRouter):
        return tuple(
            registration.provider.provider_id for registration in model.candidates
        )
    if model is None:
        return ()
    return (model.provider_id,)


def _safe_followup_failure_code(error: Exception) -> str:
    candidate = getattr(error, "code", None)
    if not isinstance(candidate, str):
        candidate = type(error).__name__
    normalized = re.sub(r"[^a-z0-9_]+", "_", candidate.casefold()).strip("_")
    return f"followup_{normalized[:96] or 'revalidation_failed'}"


def _validate_conversation_id(value: str) -> None:
    if not isinstance(value, str) or _CONVERSATION_ID.fullmatch(value) is None:
        raise ValueError("conversation_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def _source_alias(value: str) -> str:
    return _SOURCE_ALIAS_SEPARATOR.sub("-", value.casefold()).strip("-")


def _resolve_source_selector(
    selector: str,
    sources: tuple[SourceRegistration, ...],
) -> SourceRegistration:
    if not isinstance(selector, str) or not (candidate := selector.strip()):
        raise SourceSelectionError("Source selector must be non-empty terminal text.")
    try:
        encoded_length = len(candidate.encode("utf-8"))
    except UnicodeEncodeError:
        raise SourceSelectionError(
            "Source selector must be non-empty terminal text."
        ) from None
    if encoded_length > 1_024 or any(
        ord(character) < 32 or ord(character) == 127 for character in candidate
    ):
        raise SourceSelectionError("Source selector must be non-empty terminal text.")
    exact_id = tuple(source for source in sources if source.id == candidate)
    if exact_id:
        return exact_id[0]
    folded = candidate.casefold()
    matches = {
        source.id: source
        for source in sources
        if source.display_name.casefold() == folded
        or _source_alias(source.display_name) == folded
    }
    if not matches:
        raise SourceSelectionError(
            "No active source matches that name. Use /sources to list choices."
        )
    if len(matches) > 1:
        raise SourceSelectionError(
            "That source name is ambiguous. Use /source to choose one."
        )
    return next(iter(matches.values()))


class AgentHomeError(AgentError):
    """Base failure for isolated local agent-home admission."""


class AgentNameError(AgentHomeError, ValueError):
    pass


class AgentAlreadyExistsError(AgentHomeError):
    pass


class AgentNotFoundError(AgentHomeError):
    pass


class AgentIdentityMismatchError(AgentHomeError):
    pass


class AgentModelConfigurationError(AgentHomeError):
    """A saved model route whose profile facts can no longer be admitted."""


class HostActiveError(AgentHomeError):
    code = "host_active"


class AgentNotConfiguredError(AgentHomeError):
    pass


class SourceSelectionError(AgentError, ValueError):
    """A requested source is missing, ambiguous, or outside current admission."""


@dataclass(frozen=True, slots=True)
class SourceEditPreview:
    """Safe bounded facts shown immediately before a source connection edit."""

    current_source_id: str
    replacement_source_id: str
    display_name: str
    adapter_id: str
    resource_count: int
    relationship_count: int
    read_mode: SourceReadMode
    preserved_read_resource_count: int
    omitted_read_resources: tuple[str, ...]

    @property
    def identity_changed(self) -> bool:
        return self.current_source_id != self.replacement_source_id


@dataclass(frozen=True, slots=True)
class SourceEditResult:
    """Committed source connection edit plus best-effort credential cleanup."""

    source: SourceRegistration
    identity_changed: bool
    previous_credential_deleted: bool


SourceEditConfirmationHandler = Callable[[SourceEditPreview], Awaitable[bool]]


class _WriterLock:
    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self._descriptor = descriptor

    @classmethod
    def acquire(cls, path: Path) -> _WriterLock:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise AgentHomeError(f"cannot open agent writer lock: {path}") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise AgentHomeError("agent writer lock must be a regular file")
            import fcntl

            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                if error.errno in {errno.EACCES, errno.EAGAIN}:
                    raise HostActiveError(
                        f"host_active: another writer owns {path.parent.parent}"
                    ) from error
                raise
            os.ftruncate(descriptor, 0)
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            os.fsync(descriptor)
            return cls(path, descriptor)
        except BaseException:
            os.close(descriptor)
            raise

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor < 0:
            return
        self._descriptor = -1
        import fcntl

        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class EmbeddedAgent:
    """Identity, catalog, tools, transcript loop, and nothing else."""

    def __init__(
        self,
        *,
        identity: AgentIdentity,
        home: Path,
        writer_lock: _WriterLock,
        store: SQLiteStateStore,
        loop: AgentLoop | None,
        transcripts: SQLiteStateStore,
        capabilities: CapabilityRegistry,
        catalog_service: CatalogService,
        data_view: CatalogDataView,
        capability_runtime: CapabilityRuntime,
        job_owner: JobOwner,
        job_supervisor: JobSupervisor,
        data_profile_admission: DataProfileAdmission,
        followup_wake: asyncio.Event,
        followup_model_routes: tuple[str, ...],
        data_profile_job_domain: DataProfileCapabilityDomain,
        learning_candidate_guard: LearningCandidateGuard,
        semantic_domain: SemanticCapabilityDomain,
        postgresql_update_backend: PostgreSQLUpdatePreviewBackend,
        memory_store: MemoryStore,
        skill_store: SkillStore,
        candidate_reviewer: OneShotCandidateReviewer,
        data_context_builder: DataContextBuilder | None,
        artifact_store: AgentHomeArtifactStore,
        artifact_delivery: LocalArtifactDelivery,
        candidate_acceptance_supported: bool,
        mutation_lock: asyncio.Lock,
        model_profile: ModelProfile | None,
        model_route: ModelRoute | None,
        limits: LoopLimits,
        secret_provider: SecretProvider,
        keychain: CredentialSession,
        owns_credential_session: bool,
        model_validator: ModelProvider | None,
        mcp_client_factory: MCPClientFactory,
        mcp_activated_bindings: tuple[MCPActivatedBinding, ...],
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        self.identity = identity
        self.home = home
        self.model_profile = model_profile
        self.model_route = model_route
        self._limits = limits
        self._secret_provider = secret_provider
        self._keychain = keychain
        self._owns_credential_session = owns_credential_session
        self._model_validator = model_validator
        self._mcp_client_factory = mcp_client_factory
        self._mcp_activated_bindings = {
            item.binding.binding_id: item for item in mcp_activated_bindings
        }
        self._writer_lock = writer_lock
        self._store = store
        self._loop = loop
        self._transcripts = transcripts
        self._capabilities = capabilities
        self._catalog_service = catalog_service
        self._data_view = data_view
        self._capability_runtime = capability_runtime
        self._job_owner = job_owner
        self._job_supervisor = job_supervisor
        self._data_profile_admission = data_profile_admission
        self._followup_wake = followup_wake
        self._followup_model_routes = followup_model_routes
        self._followup_driver: asyncio.Task[None] | None = None
        self._data_profile_job_domain = data_profile_job_domain
        self._learning_candidate_guard = learning_candidate_guard
        self._semantic_domain = semantic_domain
        self._postgresql_update_backend = postgresql_update_backend
        self._memory_store = memory_store
        self._skill_store = skill_store
        self._candidate_reviewer = candidate_reviewer
        self._data_context_builder = data_context_builder
        self._artifact_store = artifact_store
        self._artifact_delivery = artifact_delivery
        self._candidate_acceptance_supported = candidate_acceptance_supported
        self._clock = clock
        self._id_factory = id_factory
        self._mutation_lock = mutation_lock
        self._run_lock = asyncio.Lock()
        self._source_permission_confirmation_key = secrets.token_bytes(32)
        self._source_permission_previews: dict[str, SourcePermissionsPreview] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._model_reopen_required = False

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
    async def delete(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        keychain: KeychainStore | None = None,
    ) -> None:
        """Delete one inactive agent home after removing its owned credentials."""

        _, cancelled = await _await_async_completion(
            lambda: cls._delete_admitted_home(
                name,
                root=root,
                keychain=keychain or KeychainSecretProvider(),
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
    ) -> None:
        del cls
        (home, writer_lock), _cancelled = await _await_sync_completion(
            lambda: _admit_agent_home(name, root, False)
        )
        store: SQLiteStateStore | None = None
        try:
            manifest, _cancelled = await _await_sync_completion(
                lambda: _read_manifest(home, name)
            )
            store = SQLiteStateStore(home / "state.db")
            identity = await store.load_identity()
            if identity is None or identity != manifest:
                raise AgentIdentityMismatchError(
                    "agent.toml does not match state.db identity"
                )
            model_document, _cancelled = await _await_sync_completion(
                lambda: _read_model_configuration_document(home)
            )
            sources = await store.list_sources(identity.id)
            references = _owned_agent_credential_references(
                identity.id,
                model_document=model_document,
                sources=sources,
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
        keychain: KeychainStore | None = None,
        model_validator: ModelProvider | None = None,
        reviewer_model: ModelProvider | None = None,
        reviewer_profile: ModelProfile | None = None,
        reviewer_max_estimated_cost_usd: Decimal | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        downloads_directory: Path | None = None,
        connected_job_profiles: tuple[ConnectedJobProfile, ...] = (),
    ) -> Self:
        if downloads_directory is not None and not isinstance(
            downloads_directory, Path
        ):
            raise TypeError("downloads_directory must be pathlib.Path or None")
        resolved_clock = clock or _utc_now
        resolved_ids = id_factory or _new_id
        credential_session, owns_credential_session = _resolve_credential_session(
            keychain
        )
        runtime_secrets = secret_provider or credential_session
        model, model_profile, model_route, limits = _resolve_configuration(
            config,
            model=model,
            model_profile=model_profile,
            limits=limits,
            secret_provider=runtime_secrets,
        )
        _validate_custom_loop(model, model_profile, context_builder, tools)
        (home, writer_lock), cancelled = await _await_sync_completion(
            lambda: _admit_agent_home(name, root, True)
        )
        if cancelled:
            writer_lock.release()
            raise asyncio.CancelledError
        store: SQLiteStateStore | None = None
        published = False
        try:
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
                references=store,
                clock=resolved_clock,
                id_factory=resolved_ids,
            )
            artifact_delivery = await LocalArtifactDelivery.open(
                agent_id=identity.id,
                agent_home=home,
                artifacts=artifact_store,
                sources=store,
                downloads_directory=downloads_directory,
                clock=resolved_clock,
                id_factory=resolved_ids,
            )
            embedded = await cls._compose(
                identity=identity,
                home=home,
                writer_lock=writer_lock,
                store=store,
                model=model,
                model_profile=model_profile,
                model_route=model_route,
                context_builder=context_builder,
                tools=tools,
                limits=limits,
                clock=resolved_clock,
                id_factory=resolved_ids,
                secret_provider=runtime_secrets,
                mcp_client_factory=mcp_client_factory,
                keychain=credential_session,
                owns_credential_session=owns_credential_session,
                model_validator=model_validator,
                reviewer_model=reviewer_model,
                reviewer_profile=reviewer_profile,
                reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
                observer=observer,
                approval_handler=approval_handler,
                artifact_store=artifact_store,
                artifact_delivery=artifact_delivery,
                connected_job_profiles=connected_job_profiles,
            )
            _, cancelled = await _await_sync_completion(
                lambda: _write_manifest(home, identity)
            )
            published = True
            if cancelled:
                raise asyncio.CancelledError
            return embedded
        except BaseException:
            try:
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
        keychain: KeychainStore | None = None,
        model_validator: ModelProvider | None = None,
        reviewer_model: ModelProvider | None = None,
        reviewer_profile: ModelProfile | None = None,
        reviewer_max_estimated_cost_usd: Decimal | None = None,
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
        downloads_directory: Path | None = None,
        connected_job_profiles: tuple[ConnectedJobProfile, ...] = (),
    ) -> Self:
        if downloads_directory is not None and not isinstance(
            downloads_directory, Path
        ):
            raise TypeError("downloads_directory must be pathlib.Path or None")
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
            model, model_profile, model_route, limits = _resolve_configuration(
                config,
                model=model,
                model_profile=model_profile,
                limits=limits,
                secret_provider=runtime_secrets,
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
        store: SQLiteStateStore | None = None
        try:
            manifest, cancelled = await _await_sync_completion(
                lambda: _read_manifest(home, name)
            )
            if cancelled:
                raise asyncio.CancelledError
            store = await SQLiteStateStore.open(home / "state.db", clock=resolved_clock)
            identity = await store.load_identity()
            if identity is None or identity != manifest:
                raise AgentIdentityMismatchError(
                    "agent.toml does not match state.db identity"
                )
            _, cancelled = await _await_async_completion(
                lambda: store.recover_unfinished_runs(
                    identity.id,
                    created_at=resolved_clock(),
                )
            )
            if cancelled:
                raise asyncio.CancelledError
            if not explicit_configuration:
                persisted, cancelled = await _await_sync_completion(
                    lambda: _read_model_configuration(home, identity.id)
                )
                if cancelled:
                    raise asyncio.CancelledError
                model, model_profile, model_route, limits = _resolve_configuration(
                    persisted,
                    model=None,
                    model_profile=None,
                    limits=limit_override,
                    secret_provider=runtime_secrets,
                )
            artifact_store = await AgentHomeArtifactStore.open(
                agent_id=identity.id,
                agent_home=home,
                references=store,
                clock=resolved_clock,
                id_factory=resolved_ids,
            )
            artifact_delivery = await LocalArtifactDelivery.open(
                agent_id=identity.id,
                agent_home=home,
                artifacts=artifact_store,
                sources=store,
                downloads_directory=downloads_directory,
                clock=resolved_clock,
                id_factory=resolved_ids,
            )
            return await cls._compose(
                identity=identity,
                home=home,
                writer_lock=writer_lock,
                store=store,
                model=model,
                model_profile=model_profile,
                model_route=model_route,
                context_builder=context_builder,
                tools=tools,
                limits=limits,
                clock=resolved_clock,
                id_factory=resolved_ids,
                secret_provider=runtime_secrets,
                mcp_client_factory=mcp_client_factory,
                keychain=credential_session,
                owns_credential_session=owns_credential_session,
                model_validator=model_validator,
                reviewer_model=reviewer_model,
                reviewer_profile=reviewer_profile,
                reviewer_max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
                observer=observer,
                approval_handler=approval_handler,
                artifact_store=artifact_store,
                artifact_delivery=artifact_delivery,
                connected_job_profiles=connected_job_profiles,
            )
        except BaseException:
            if store is not None:
                await store.close()
            writer_lock.release()
            raise

    @classmethod
    async def _compose(
        cls,
        *,
        identity: AgentIdentity,
        home: Path,
        writer_lock: _WriterLock,
        store: SQLiteStateStore,
        model: ModelProvider | None,
        model_profile: ModelProfile | None,
        model_route: ModelRoute | None,
        context_builder: ContextBuilder | None,
        tools: ToolRuntime | None,
        limits: LoopLimits,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        secret_provider: SecretProvider,
        mcp_client_factory: MCPClientFactory | None,
        keychain: CredentialSession,
        owns_credential_session: bool,
        model_validator: ModelProvider | None,
        reviewer_model: ModelProvider | None,
        reviewer_profile: ModelProfile | None,
        reviewer_max_estimated_cost_usd: Decimal | None,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
        artifact_store: AgentHomeArtifactStore,
        artifact_delivery: LocalArtifactDelivery,
        connected_job_profiles: tuple[ConnectedJobProfile, ...],
    ) -> Self:
        catalog_service = CatalogService(store, store)
        data_view = CatalogDataView(store, catalog_service, store)
        catalog = catalog_declarations(identity.id, data_view)
        sqlite_backend = SQLiteQueryBackend(store, data_view)
        postgresql_backend = PostgreSQLQueryBackend(
            store, data_view, secret_provider or keychain
        )
        sqlite = sqlite_query_declarations(identity.id, sqlite_backend)
        postgresql = postgresql_query_declarations(
            identity.id,
            postgresql_backend,
        )
        postgresql_preview_backend = PostgreSQLUpdatePreviewBackend(
            store,
            data_view,
            secret_provider or keychain,
            receipt_store=store,
            clock=clock,
        )
        postgresql_preview = postgresql_update_preview_declarations(
            identity.id,
            postgresql_preview_backend,
        )
        postgresql_update = postgresql_update_declarations(
            identity.id,
            postgresql_preview_backend,
        )
        local_file_backend = LocalDirectoryReadBackend(store, store, data_view)
        local_files = local_file_read_declarations(identity.id, local_file_backend)
        mutation_lock = asyncio.Lock()
        memory_store = MemoryStore(home, mutation_lock)
        skill_store = SkillStore(home, mutation_lock)
        resolved_reviewer_model = reviewer_model
        resolved_reviewer_profile = reviewer_profile
        if (
            resolved_reviewer_model is None
            and resolved_reviewer_profile is None
            and reviewer_max_estimated_cost_usd is not None
            and model_route is not None
        ):
            (
                resolved_reviewer_model,
                resolved_reviewer_profile,
            ) = _candidate_reviewer_from_route(
                model_route,
                secret_provider=secret_provider or keychain,
            )
        resolved_reviewer_profile = _resolve_candidate_reviewer_profile(
            resolved_reviewer_model,
            resolved_reviewer_profile,
        )
        candidate_reviewer = OneShotCandidateReviewer(
            agent_id=identity.id,
            store=store,
            memory=memory_store,
            skills=skill_store,
            catalog=data_view,
            model=resolved_reviewer_model,
            profile=resolved_reviewer_profile,
            max_estimated_cost_usd=reviewer_max_estimated_cost_usd,
            clock=clock,
        )
        memory = memory_set_declarations(memory_store)
        skills = skill_declarations(skill_store)
        semantics = semantic_declarations(identity.id, store)
        artifacts = (
            artifact_declarations(
                artifact_delivery,
                artifact_store,
                agent_id=identity.id,
                local_file_backend=local_file_backend,
                sqlite_backend=sqlite_backend,
                postgresql_backend=postgresql_backend,
                clock=clock,
            )
            if artifact_store.available
            else None
        )
        job_owner = JobOwner(
            agent_id=identity.id,
            store=store,
            connected_profiles=connected_job_profiles,
            clock=clock,
            id_factory=id_factory,
        )
        job_lifecycle = job_capability_declarations(job_owner)
        data_profile_jobs, data_profile_admission = data_profile_declarations(
            agent_id=identity.id,
            catalog=data_view,
            owner=job_owner,
            sqlite_backend=sqlite_backend,
            postgresql_backend=postgresql_backend,
            local_file_backend=local_file_backend,
            clock=clock,
        )
        learning_candidate_guard = LearningCandidateGuard()
        data_declarations = CapabilityDeclarations(
            domain_owner_id=DATA_DOMAIN_OWNER_ID,
            capabilities=(
                *catalog.capabilities,
                *sqlite.capabilities,
                *postgresql.capabilities,
                *postgresql_preview.capabilities,
                *postgresql_update.capabilities,
                *local_files.capabilities,
            ),
            executor_ids=tuple(
                item.executor_id
                for item in (
                    *catalog.capabilities,
                    *sqlite.capabilities,
                    *postgresql.capabilities,
                    *postgresql_preview.capabilities,
                    *postgresql_update.capabilities,
                    *local_files.capabilities,
                )
            ),
            tool_views=(
                *catalog.tool_views,
                *sqlite.tool_views,
                *postgresql.tool_views,
                *postgresql_preview.tool_views,
                *postgresql_update.tool_views,
                *local_files.tool_views,
            ),
        )
        memory_declarations = CapabilityDeclarations(
            domain_owner_id=MEMORY_DOMAIN_OWNER_ID,
            capabilities=memory.capabilities,
            executor_ids=tuple(item.executor_id for item in memory.capabilities),
            tool_views=memory.tool_views,
        )
        skill_declaration_bundle = CapabilityDeclarations(
            domain_owner_id=SKILL_DOMAIN_OWNER_ID,
            capabilities=skills.capabilities,
            executor_ids=tuple(item.executor_id for item in skills.capabilities),
            tool_views=skills.tool_views,
        )
        semantic_declaration_bundle = CapabilityDeclarations(
            domain_owner_id=SEMANTIC_DOMAIN_OWNER_ID,
            capabilities=semantics.capabilities,
            executor_ids=tuple(item.executor_id for item in semantics.capabilities),
            tool_views=semantics.tool_views,
        )
        artifact_declaration_bundle = (
            None
            if artifacts is None
            else CapabilityDeclarations(
                domain_owner_id=ARTIFACT_DOMAIN_OWNER_ID,
                capabilities=artifacts.capabilities,
                executor_ids=tuple(item.executor_id for item in artifacts.capabilities),
                tool_views=artifacts.tool_views,
            )
        )
        job_declaration_bundle = CapabilityDeclarations(
            domain_owner_id=JOB_DOMAIN_OWNER_ID,
            capabilities=job_lifecycle.capabilities,
            executor_ids=tuple(item.executor_id for item in job_lifecycle.capabilities),
            tool_views=job_lifecycle.tool_views,
        )
        data_profile_declaration_bundle = CapabilityDeclarations(
            domain_owner_id=DATA_PROFILE_DOMAIN_OWNER_ID,
            capabilities=data_profile_jobs.capabilities,
            executor_ids=tuple(
                item.executor_id for item in data_profile_jobs.capabilities
            ),
            tool_views=data_profile_jobs.tool_views,
        )
        data_domain = DataCapabilityDomain(
            data_declarations,
            data_view,
            learning_candidate_guard,
        )
        memory_domain = MemoryCapabilityDomain(
            memory_declarations,
            learning_candidate_guard,
        )
        skill_domain = SkillCapabilityDomain(
            skill_declaration_bundle,
            learning_candidate_guard,
        )
        semantic_domain = SemanticCapabilityDomain(
            semantic_declaration_bundle,
            data_view,
            store,
            learning_candidate_guard,
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
                learning_candidate_guard,
            )
        )
        job_domain = JobCapabilityDomain(job_declaration_bundle, job_owner)
        data_profile_job_domain = DataProfileCapabilityDomain(
            data_profile_declaration_bundle,
            catalog=data_view,
            admission=data_profile_admission,
            learning=learning_candidate_guard,
        )
        if model is not None and context_builder is None:
            assert model_profile is not None
            if not model_profile.supports_tools:
                raise AgentNotConfiguredError(
                    "the data agent requires a tool-capable model profile"
                )
        resolved_mcp_client_factory = (
            mcp_client_factory or StreamableHTTPMCPClientFactory()
        )
        mcp_domain, mcp_activated_bindings, mcp_executors = await activate_mcp_domain(
            agent_id=identity.id,
            store=store,
            client_factory=resolved_mcp_client_factory,
            secrets=secret_provider,
            clock=clock,
        )
        domains = (
            data_domain,
            memory_domain,
            skill_domain,
            semantic_domain,
            job_domain,
            data_profile_job_domain,
            *((artifact_domain,) if artifact_domain is not None else ()),
            *((mcp_domain,) if mcp_domain is not None else ()),
        )
        capabilities = CapabilityRegistry(
            declarations=tuple(domain.declarations for domain in domains),
            executors=(
                *catalog.executors,
                *sqlite.executors,
                *postgresql.executors,
                *postgresql_preview.executors,
                *postgresql_update.executors,
                *local_files.executors,
                *memory.executors,
                *skills.executors,
                *semantics.executors,
                *job_lifecycle.executors,
                *data_profile_jobs.executors,
                *(artifacts.executors if artifacts is not None else ()),
                *mcp_executors,
            ),
        )
        capability_runtime = CapabilityRuntime(
            capabilities,
            domains,
            approval_handler=approval_handler,
            mutation_lock=mutation_lock,
            observer=observer,
            clock=clock,
            artifacts=artifact_store,
            limits=limits,
        )
        followup_wake = asyncio.Event()
        job_supervisor = JobSupervisor(
            agent_id=identity.id,
            store=store,
            owner=job_owner,
            runtime=capability_runtime,
            revalidate_external=data_profile_admission.revalidate_external,
            artifacts=artifact_store,
            clock=clock,
            id_factory=id_factory,
            on_terminal=lambda job: (
                followup_wake.set()
                if job.specification.execution_mode is JobExecutionMode.DAITA
                else None
            ),
        )
        job_owner.bind_wake(job_supervisor.wake)
        resolved_context = context_builder
        resolved_tools = tools
        if model is not None and resolved_context is None:
            assert model_profile is not None
            resolved_context = DataContextBuilder(
                data_view,
                profile=model_profile,
                memory=memory_store,
                skills=skill_store,
                semantics=store,
                artifact_destinations=(
                    artifact_delivery if artifacts is not None else None
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
                clock=clock,
                observer=observer,
                stream_model_calls=(
                    model_profile.supports_streaming
                    if model_profile is not None
                    else False
                ),
            )
        )
        embedded = cls(
            identity=identity,
            home=home,
            writer_lock=writer_lock,
            store=store,
            loop=loop,
            transcripts=transcripts,
            capabilities=capabilities,
            catalog_service=catalog_service,
            data_view=data_view,
            capability_runtime=capability_runtime,
            job_owner=job_owner,
            job_supervisor=job_supervisor,
            data_profile_admission=data_profile_admission,
            followup_wake=followup_wake,
            followup_model_routes=_stage_c_model_routes(model, model_route),
            data_profile_job_domain=data_profile_job_domain,
            learning_candidate_guard=learning_candidate_guard,
            semantic_domain=semantic_domain,
            postgresql_update_backend=postgresql_preview_backend,
            memory_store=memory_store,
            skill_store=skill_store,
            candidate_reviewer=candidate_reviewer,
            data_context_builder=(
                resolved_context
                if isinstance(resolved_context, DataContextBuilder)
                else None
            ),
            artifact_store=artifact_store,
            artifact_delivery=artifact_delivery,
            candidate_acceptance_supported=(
                isinstance(resolved_context, DataContextBuilder)
                and resolved_tools is capability_runtime
            ),
            mutation_lock=mutation_lock,
            model_profile=model_profile,
            model_route=model_route,
            limits=limits,
            secret_provider=secret_provider,
            keychain=keychain,
            owns_credential_session=owns_credential_session,
            model_validator=model_validator,
            mcp_client_factory=resolved_mcp_client_factory,
            mcp_activated_bindings=mcp_activated_bindings,
            clock=clock,
            id_factory=id_factory,
        )
        await job_supervisor.start()
        embedded._start_followup_driver()
        return embedded

    def model_requires_explicit_limits(self, *, provider: str, model: str) -> bool:
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
        if provider != "codex":
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
        """

        provider_name, model_name, endpoint, requires_credential = (
            _admit_model_selection(
                provider,
                model,
                base_url,
            )
        )
        requires_subscription_credential = (
            provider_name in _SUBSCRIPTION_CREDENTIAL_PROVIDERS
        )
        if requires_subscription_credential:
            if api_key is not None:
                raise ValueError("Codex subscription login does not accept an API key")
            if (
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

        async with self._mutation_lock:
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
            if requires_credential:
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
            replacement = AgentConfig(model_route=route, limits=self._limits)
            committed = False
            try:
                if reference is not None:
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
                if reference is not None and not committed:
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
        source_id: str | None = None,
        job_executor_profile_id: str | None = None,
    ) -> LoopExit:
        return await self._run(
            message,
            conversation_id=conversation_id,
            source_id=source_id,
            job_executor_profile_id=job_executor_profile_id,
        )

    async def learn(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_id: str | None = None,
    ) -> LoopExit:
        """Run one explicit user-authorized foreground learning action."""

        return await self._run(
            message,
            conversation_id=conversation_id,
            source_id=source_id,
            explicit_learning=True,
        )

    async def _run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
        source_id: str | None = None,
        learning_candidate_id: str | None = None,
        learning_candidate_text: str | None = None,
        learning_candidate: LearningCandidate | None = None,
        explicit_learning: bool = False,
        job_executor_profile_id: str | None = None,
    ) -> LoopExit:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if source_id is not None and (
            not isinstance(source_id, str) or not source_id.strip()
        ):
            raise ValueError("source_id must be a non-empty string or None")
        if job_executor_profile_id is not None and (
            not isinstance(job_executor_profile_id, str)
            or not job_executor_profile_id.strip()
        ):
            raise ValueError(
                "job_executor_profile_id must be a non-empty string or None"
            )
        if learning_candidate_id is not None and (
            not isinstance(learning_candidate_id, str)
            or not learning_candidate_id.strip()
        ):
            raise ValueError("learning_candidate_id must be a non-empty string or None")
        if not (
            (learning_candidate_id is None)
            == (learning_candidate_text is None)
            == (learning_candidate is None)
        ):
            raise ValueError(
                "learning candidate guard, id, and rendered content must be set together"
            )
        if self._model_reopen_required:
            raise AgentNotConfiguredError(
                "model configuration changed; close and reopen required"
            )
        loop = self._require_loop()
        async with self._run_lock:
            return await self._run_locked(
                loop,
                message,
                conversation_id=conversation_id,
                source_id=source_id,
                learning_candidate_id=learning_candidate_id,
                learning_candidate_text=learning_candidate_text,
                learning_candidate=learning_candidate,
                explicit_learning=explicit_learning,
                job_executor_profile_id=job_executor_profile_id,
            )

    async def _run_locked(
        self,
        loop: AgentLoop,
        message: str,
        *,
        conversation_id: str | None,
        source_id: str | None,
        learning_candidate_id: str | None,
        learning_candidate_text: str | None,
        learning_candidate: LearningCandidate | None,
        explicit_learning: bool = False,
        run_id: str | None = None,
        job_executor_profile_id: str | None = None,
    ) -> LoopExit:
        """Run once while the caller owns the foreground lifecycle lock."""

        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        if source_id is not None and (
            not isinstance(source_id, str) or not source_id.strip()
        ):
            raise ValueError("source_id must be a non-empty string or None")
        if self._model_reopen_required:
            raise AgentNotConfiguredError(
                "model configuration changed; close and reopen required"
            )
        self._require_open()
        supplied_conversation = conversation_id is not None
        resolved_conversation = (
            self._id_factory("conversation")
            if conversation_id is None
            else conversation_id
        )
        _validate_conversation_id(resolved_conversation)
        conversation_exists, conversation, older_history_exists = (
            await self._store.completed_conversation_tail(
                self.identity.id,
                resolved_conversation,
            )
        )
        if supplied_conversation and not conversation_exists:
            raise ValueError("unknown conversation for this agent")
        if not supplied_conversation and conversation_exists:
            raise ValueError("generated conversation id already exists")
        conversation_source_id = (
            await self._store.conversation_source_id(
                self.identity.id,
                resolved_conversation,
            )
            if supplied_conversation
            else await self._store.load_active_source_id(self.identity.id)
        )
        active_sources = tuple(
            source
            for source in await self._store.list_sources(self.identity.id)
            if source.active
        )
        if conversation_source_id is None and len(active_sources) == 1:
            conversation_source_id = active_sources[0].id
        effective_source_id = (
            source_id.strip() if source_id is not None else conversation_source_id
        )
        if active_sources and effective_source_id is None:
            raise SourceSelectionError(
                "Multiple data sources are attached. Select one with /source "
                "before asking a question."
            )
        if effective_source_id is not None and not any(
            source.id == effective_source_id for source in active_sources
        ):
            raise SourceSelectionError(
                "The selected data source is not active for this agent."
            )
        scoped_conversation = tuple(
            item
            for item in conversation
            if item.transcript.run.source_id == effective_source_id
        )
        prior_messages = _project_completed_history(
            scoped_conversation,
            older_history_exists=(
                older_history_exists or len(scoped_conversation) < len(conversation)
            ),
        )
        run_input = RunInput(
            id=run_id or self._id_factory("run"),
            agent_id=self.identity.id,
            message=message.strip(),
            created_at=self._clock(),
            conversation_id=resolved_conversation,
            source_id=effective_source_id,
            conversation_source_id=conversation_source_id,
        )
        if job_executor_profile_id is not None:
            self._data_profile_job_domain.select_connected_executor(
                run_input.id,
                job_executor_profile_id.strip(),
            )
        if learning_candidate_id is not None:
            if self._data_context_builder is None:
                raise AgentHomeError(
                    "learning candidate acceptance requires DataContextBuilder"
                )
            self._data_context_builder.select_learning_candidate(
                run_input.id,
                learning_candidate_id,
                cast(str, learning_candidate_text),
            )
            self._learning_candidate_guard.select(
                run_input.id,
                cast(LearningCandidate, learning_candidate),
            )
        if explicit_learning:
            self._semantic_domain.select_explicit_learning_run(run_input.id)
        try:
            return await loop.run(
                run_input,
                prior_messages=prior_messages,
            )
        finally:
            self._artifact_delivery.end_run(run_input.id)
            if learning_candidate_id is not None:
                assert self._data_context_builder is not None
                self._data_context_builder.clear_learning_candidate(run_input.id)
                self._learning_candidate_guard.clear(run_input.id)
            if explicit_learning:
                self._semantic_domain.clear_explicit_learning_run(run_input.id)
            if job_executor_profile_id is not None:
                self._data_profile_job_domain.clear_connected_executor(run_input.id)

    async def transcript(self, run_id: str) -> Transcript:
        self._require_open()
        return await self._transcripts.load(run_id)

    def _start_followup_driver(self) -> None:
        if (
            self._loop is None
            or self._data_context_builder is None
            or self._limits.max_estimated_cost_usd is None
            or not self._followup_model_routes
        ):
            return
        if self._followup_driver is not None:
            raise RuntimeError("autonomous follow-up driver is already started")
        self._followup_driver = asyncio.create_task(
            self._drive_followups(),
            name=f"daita-followups:{self.identity.id}",
        )

    async def _drive_followups(self) -> None:
        try:
            while not self._closed:
                self._followup_wake.clear()
                try:
                    await self._store.recover_stale_autonomous_followups(
                        self.identity.id,
                        recovered_at=self._clock(),
                    )
                    await self._admit_terminal_followups()
                    await self._reconcile_terminal_followups()
                    claimed = await self._store.claim_next_autonomous_followup(
                        self.identity.id,
                        claim_token=self._id_factory("followup-claim"),
                        reserved_run_id=self._id_factory("run"),
                        claimed_at=self._clock(),
                        lease_seconds=FOLLOWUP_LEASE_SECONDS,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    claimed = None
                if claimed is not None:
                    try:
                        await self._execute_followup(claimed.followup_id)
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        # The durable claim or terminal run remains recoverable. One
                        # item must not stop unrelated Stage C progress.
                        pass
                    continue
                timeout = _STAGE_C_SAFETY_WAKE_SECONDS
                try:
                    deadline = await self._store.next_autonomous_followup_deadline(
                        self.identity.id
                    )
                    if deadline is not None:
                        timeout = min(
                            timeout,
                            max(0.25, (deadline - self._clock()).total_seconds()),
                        )
                except Exception:
                    pass
                try:
                    await asyncio.wait_for(
                        self._followup_wake.wait(),
                        timeout=timeout,
                    )
                except TimeoutError:
                    pass
        except asyncio.CancelledError:
            return

    async def _admit_terminal_followups(self) -> None:
        if self._loop is None or self._limits.max_estimated_cost_usd is None:
            return
        routes = self._followup_model_routes
        capability_ids = _stage_c_capability_ids(self._capabilities)
        jobs = await self._store.list_unbound_terminal_daita_jobs(self.identity.id)
        for job in jobs:
            try:
                followup = create_terminal_job_followup(
                    job,
                    followup_id=self._id_factory("followup"),
                    grant_id=self._id_factory("grant"),
                    scope_id=self._id_factory("scope"),
                    received_at=self._clock(),
                    allowed_capability_ids=capability_ids,
                    eligible_model_routes=routes,
                    limits=self._limits,
                )
                await self._store.admit_autonomous_followup(followup)
            except asyncio.CancelledError:
                raise
            except Exception:
                continue

    async def _reconcile_terminal_followups(self) -> None:
        active = await self._store.list_autonomous_followups(
            self.identity.id,
            dispositions=frozenset(
                {
                    FollowupDisposition.RUNNING,
                    FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION,
                }
            ),
        )
        for followup in active:
            try:
                if followup.disposition is FollowupDisposition.RUNNING:
                    assert followup.reserved_run_id is not None
                    try:
                        result = await self._store.result(followup.reserved_run_id)
                    except KeyError:
                        continue
                    if result is None:
                        continue
                    followup = (
                        await self._store.mark_autonomous_followup_run_terminal(
                            self.identity.id,
                            followup.followup_id,
                            run_id=followup.reserved_run_id,
                            terminal_at=result.created_at,
                        )
                        or followup
                    )
                if (
                    followup.disposition
                    is FollowupDisposition.RUN_TERMINAL_PENDING_FINALIZATION
                ):
                    await self._store.finalize_autonomous_followup(
                        self.identity.id,
                        followup.followup_id,
                        delivery_id=self._id_factory("delivery"),
                        finalized_at=self._clock(),
                    )
            except asyncio.CancelledError:
                raise
            except Exception:
                continue

    async def _execute_followup(self, followup_id: str) -> None:
        loop = self._require_loop()
        async with self._run_lock:
            followup = await self._store.load_autonomous_followup(
                self.identity.id,
                followup_id,
            )
            if (
                followup is None
                or followup.disposition is not FollowupDisposition.CLAIMED
                or followup.claim_token is None
                or followup.reserved_run_id is None
            ):
                return
            try:
                job = await self._revalidate_followup(followup)
                _conversation_exists, conversation, older_history_exists = (
                    await self._store.completed_conversation_tail(
                        self.identity.id,
                        followup.conversation_id,
                    )
                )
                scoped_conversation = tuple(
                    item
                    for item in conversation
                    if item.transcript.run.source_id
                    in followup.execution_scope.allowed_source_ids
                )
                prior_messages = _project_completed_history(
                    scoped_conversation,
                    older_history_exists=(
                        older_history_exists
                        or len(scoped_conversation) < len(conversation)
                    ),
                )
                run_input = RunInput(
                    id=followup.reserved_run_id,
                    agent_id=self.identity.id,
                    message=FOLLOWUP_INSTRUCTION,
                    created_at=self._clock(),
                    conversation_id=followup.conversation_id,
                    source_id=(job.source_ids[0] if len(job.source_ids) == 1 else None),
                    conversation_source_id=(
                        job.source_ids[0] if len(job.source_ids) == 1 else None
                    ),
                    start=RunStartEnvelope(
                        origin=RunOrigin.JOB_EVENT,
                        trusted_instruction_id=FOLLOWUP_INSTRUCTION_ID,
                        trusted_instruction=FOLLOWUP_INSTRUCTION,
                        instruction_digest=FOLLOWUP_INSTRUCTION_DIGEST,
                        untrusted_payload=followup.event_payload,
                        payload_digest=followup.payload_digest,
                        execution_scope=followup.execution_scope,
                    ),
                )
                prepared = await loop.prepare(
                    run_input,
                    prior_messages=prior_messages,
                )
                snapshot = prepared.context_snapshot
                audit_method = getattr(snapshot, "audit_context", None)
                if not callable(audit_method):
                    raise ValueError("followup_audit_context_unavailable")
                audit = FrozenJsonObject.from_mapping(
                    {
                        "job_id": job.job_id,
                        "job_terminal_revision": followup.job_terminal_revision,
                        "job_current_revision": job.revision,
                        "event_id": followup.event_id,
                        "payload_digest": followup.payload_digest,
                        "grant_id": followup.grant.grant_id,
                        "execution_scope_digest": followup.execution_scope.digest,
                        "capability_registry_digest": self._capabilities.digest,
                        "prepared_context": audit_method(),
                    }
                )
                bound = await self._store.bind_autonomous_followup_run(
                    self.identity.id,
                    followup.followup_id,
                    claim_token=followup.claim_token,
                    run_id=run_input.id,
                    bound_at=self._clock(),
                    audit_context=audit,
                )
                if bound is None:
                    return
            except LoopPreparationError as error:
                await self._store.fail_autonomous_followup_claim(
                    self.identity.id,
                    followup.followup_id,
                    claim_token=followup.claim_token,
                    failed_at=self._clock(),
                    failure_code=error.code,
                )
                return
            except Exception as error:
                await self._store.fail_autonomous_followup_claim(
                    self.identity.id,
                    followup.followup_id,
                    claim_token=followup.claim_token,
                    failed_at=self._clock(),
                    failure_code=_safe_followup_failure_code(error),
                )
                return
            try:
                result = await loop.run(
                    run_input,
                    prior_messages=prior_messages,
                    prepared=prepared,
                )
                await self._store.mark_autonomous_followup_run_terminal(
                    self.identity.id,
                    followup.followup_id,
                    run_id=run_input.id,
                    terminal_at=result.created_at,
                )
                await self._store.finalize_autonomous_followup(
                    self.identity.id,
                    followup.followup_id,
                    delivery_id=self._id_factory("delivery"),
                    finalized_at=self._clock(),
                )
            finally:
                self._artifact_delivery.end_run(run_input.id)

    async def _revalidate_followup(self, followup):
        if await self._store.load_identity() != self.identity:
            raise ValueError("followup_agent_identity_changed")
        current_time = self._clock()
        if followup.grant.expires_at <= current_time:
            raise ValueError("followup_grant_expired")
        if (
            followup.lease_expires_at is None
            or followup.lease_expires_at <= current_time
        ):
            raise ValueError("followup_claim_expired")
        current_capability_ceiling = frozenset(
            _stage_c_capability_ids(self._capabilities)
        )
        grant_capabilities = frozenset(followup.grant.allowed_capability_ids)
        validated_grant = self._capabilities.validate_execution_scope_grant(
            followup.grant.allowed_capability_ids,
            allowed_access_modes=followup.grant.allowed_access_modes,
            allowed_operational_effects=(followup.grant.allowed_operational_effects),
        )
        if (
            followup.grant.instruction_id != FOLLOWUP_INSTRUCTION_ID
            or followup.grant.instruction_digest != FOLLOWUP_INSTRUCTION_DIGEST
            or grant_capabilities != frozenset(validated_grant)
            or not grant_capabilities <= current_capability_ceiling
            or not followup.grant.allowed_access_modes
            <= frozenset({AccessMode.NONE, AccessMode.READ})
            or not followup.grant.allowed_operational_effects
            <= frozenset({OperationalEffect.NONE})
            or followup.execution_scope.job_id != followup.job_id
            or followup.execution_scope.job_revision != followup.job_terminal_revision
        ):
            raise ValueError("followup_grant_not_current")
        job = await self._store.load_job(self.identity.id, followup.job_id)
        if (
            job is None
            or not job.terminal
            or job.specification.execution_mode is not JobExecutionMode.DAITA
            or job.completion_binding is None
            or job.completion_binding.owner_kind
            is not JobCompletionOwnerKind.STANDALONE_FOLLOWUP
            or job.completion_binding.owner_id != followup.followup_id
            or job.completion_binding.terminal_event_id != followup.event_id
            or job.revision != followup.job_terminal_revision + 1
            or job.source_ids != followup.execution_scope.allowed_source_ids
            or job.resource_ids != followup.execution_scope.allowed_resource_ids
        ):
            raise ValueError("followup_job_binding_not_current")
        current_sensitivity = (
            job.specification.sensitivity
            if job.result is None
            else job.result.sensitivity
        )
        if (
            current_sensitivity.routing_rank
            > followup.execution_scope.sensitivity_ceiling.routing_rank
        ):
            raise ValueError("followup_sensitivity_scope_changed")
        await self._data_profile_admission.revalidate_job(job)
        return job

    async def inbox(
        self,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = 50,
    ) -> tuple[InboxItem, ...]:
        """Inspect a bounded agent-owned durable conversation inbox."""

        self._require_open()
        if conversation_id is not None:
            _validate_conversation_id(conversation_id)
        return await self._store.list_inbox(
            self.identity.id,
            conversation_id=conversation_id,
            include_acknowledged=include_acknowledged,
            limit=limit,
        )

    async def acknowledge_inbox(self, delivery_id: str) -> InboxItem | None:
        """Idempotently acknowledge one exact agent-owned inbox result."""

        self._require_open()
        if not isinstance(delivery_id, str) or not delivery_id.strip():
            raise ValueError("delivery_id must be non-empty text")
        return await self._store.acknowledge_inbox(
            self.identity.id,
            delivery_id.strip(),
            acknowledged_at=self._clock(),
        )

    async def conversation_runs(
        self,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        self._require_open()
        _validate_conversation_id(conversation_id)
        records = await self._store.conversation_runs(
            self.identity.id,
            conversation_id,
        )
        if not records:
            raise ValueError("unknown conversation for this agent")
        return records

    async def conversation_exists(self, conversation_id: str) -> bool:
        """Validate and check one bounded agent-scoped conversation identity."""

        self._require_open()
        _validate_conversation_id(conversation_id)
        return await self._store.conversation_exists(
            self.identity.id,
            conversation_id,
        )

    async def list_jobs(
        self,
        *,
        statuses: frozenset[JobStatus] = frozenset(),
        limit: int = 50,
    ) -> tuple[JobSummary, ...]:
        """Return the bounded current projection of this agent's durable jobs."""

        self._require_open()
        return await self._job_owner.list(statuses=statuses, limit=limit)

    async def inspect_job(self, job_id: str) -> JobInspection | None:
        self._require_open()
        return await self._job_owner.inspect(job_id)

    async def read_job_result(self, job_id: str) -> JobResultView | None:
        self._require_open()
        return await self._job_owner.read_result(job_id)

    async def cancel_job(self, job_id: str) -> JobInspection | None:
        self._require_open()
        return await self._job_owner.cancel(job_id)

    async def clear_conversations(self) -> int:
        """Delete transcripts and candidate records, not approved knowledge."""

        async with self._run_lock:
            self._require_open()
            cleared = await self._candidate_reviewer.clear_conversations()
            cancelled = False
            try:
                _, cancelled = await _await_async_completion(
                    self._artifact_store.remove_all_run_artifacts
                )
            except Exception:
                # SQLite deletion is already authoritative. A private orphan is
                # retried by bounded startup cleanup and never restores a ref.
                pass
            if cancelled:
                raise asyncio.CancelledError
            return cleared

    async def read_artifact(self, artifact_id: str) -> ArtifactPayload:
        self._require_open()
        return await self._artifact_store.read(artifact_id)

    async def save_artifact(
        self,
        artifact_id: str,
        destination: Path | None = None,
        *,
        filename: str | None = None,
    ) -> ArtifactDeliveryReceipt:
        async with self._mutation_lock:
            self._require_open()
            return await self._artifact_delivery.save_public(
                artifact_id,
                destination=destination,
                filename=filename,
            )

    async def export_destination(self) -> ArtifactDestination:
        async with self._mutation_lock:
            self._require_open()
            return await self._artifact_delivery.export_destination()

    async def set_export_destination(
        self,
        directory: Path,
    ) -> ArtifactDestination:
        async with self._mutation_lock:
            self._require_open()
            return await self._artifact_delivery.set_export_destination(directory)

    async def reset_export_destination(self) -> ArtifactDestination:
        async with self._mutation_lock:
            self._require_open()
            return await self._artifact_delivery.reset_export_destination()

    async def active_source(
        self,
        *,
        conversation_id: str | None = None,
    ) -> SourceRegistration | None:
        """Return the persisted default or one conversation's sticky source."""

        self._require_open()
        if conversation_id is not None:
            _validate_conversation_id(conversation_id)
            source_id = await self._store.conversation_source_id(
                self.identity.id,
                conversation_id,
            )
        else:
            source_id = await self._store.load_active_source_id(self.identity.id)
        active_sources = tuple(
            source
            for source in await self._store.list_sources(self.identity.id)
            if source.active
        )
        if source_id is None:
            return active_sources[0] if len(active_sources) == 1 else None
        return next(
            (source for source in active_sources if source.id == source_id),
            None,
        )

    async def resolve_source(self, selector: str) -> SourceRegistration:
        """Resolve one exact ID, display name, or stable display-name alias."""

        self._require_open()
        sources = tuple(
            source
            for source in await self._store.list_sources(self.identity.id)
            if source.active
        )
        return _resolve_source_selector(selector, sources)

    async def select_source(self, selector: str) -> SourceRegistration:
        """Persist one active source as the default for subsequent conversations."""

        async with self._run_lock:
            async with self._mutation_lock:
                self._require_open()
                sources = tuple(
                    source
                    for source in await self._store.list_sources(self.identity.id)
                    if source.active
                )
                selected = _resolve_source_selector(selector, sources)
                return await self._store.set_active_source_id(
                    self.identity.id,
                    selected.id,
                )

    async def read_memory(self) -> str:
        self._require_open()
        return await self._memory_store.read_memory()

    async def set_memory(self, text: str) -> None:
        self._require_open()
        await self._memory_store.set_memory(text)

    async def read_user_profile(self) -> str:
        self._require_open()
        return await self._memory_store.read_user_profile()

    async def set_user_profile(self, text: str) -> None:
        self._require_open()
        await self._memory_store.set_user_profile(text)

    async def review_learning_candidates(
        self,
        *,
        max_estimated_cost_usd: Decimal | None = None,
    ) -> LearningReviewResult:
        """Explicitly trigger one bounded auxiliary review request."""

        async with self._run_lock:
            self._require_open()
            if max_estimated_cost_usd is None or self._candidate_reviewer.enabled:
                return await self._candidate_reviewer.review(
                    max_estimated_cost_usd=max_estimated_cost_usd,
                )
            if self.model_route is None:
                return await self._candidate_reviewer.review()
            model, profile = _candidate_reviewer_from_route(
                self.model_route,
                secret_provider=self._secret_provider or self._keychain,
            )
            return await self._candidate_reviewer.review_with_model(
                model=model,
                profile=profile,
                max_estimated_cost_usd=max_estimated_cost_usd,
            )

    async def list_learning_candidates(
        self,
        *,
        status: LearningCandidateStatus | None = None,
    ) -> tuple[LearningCandidateView, ...]:
        self._require_open()
        return await self._candidate_reviewer.list_candidates(status=status)

    async def read_learning_candidate(
        self,
        candidate_id: str,
    ) -> LearningCandidateView | None:
        self._require_open()
        return await self._candidate_reviewer.read_candidate(candidate_id)

    async def edit_learning_candidate(
        self,
        candidate_id: str,
        content: LearningCandidateContent,
    ) -> LearningCandidateView:
        async with self._run_lock:
            self._require_open()
            return await self._candidate_reviewer.edit_candidate(candidate_id, content)

    async def reject_learning_candidate(
        self,
        candidate_id: str,
        reason: LearningCandidateRejectionReason,
    ) -> LearningCandidateView:
        async with self._run_lock:
            self._require_open()
            return await self._candidate_reviewer.reject_candidate(candidate_id, reason)

    async def clear_rejected_learning_candidates(self) -> int:
        async with self._run_lock:
            self._require_open()
            return await self._candidate_reviewer.clear_rejected()

    async def accept_learning_candidate(
        self,
        candidate_id: str,
        *,
        conversation_id: str | None = None,
        source_id: str | None = None,
    ) -> LoopExit:
        """Start a fresh ordinary foreground run for one selected candidate."""

        async with self._run_lock:
            self._require_open()
            if not self._candidate_acceptance_supported:
                raise AgentHomeError(
                    "learning candidate acceptance requires the built-in data "
                    "context and tool runtime"
                )
            view = await self._candidate_reviewer.read_candidate(candidate_id)
            if view is None:
                raise ValueError(f"learning candidate not found: {candidate_id}")
            if view.status is not LearningCandidateStatus.AWAITING_REVIEW:
                raise ValueError(
                    f"learning candidate is not awaiting review: {view.status.value}"
                )
            candidate = view.candidate
            effective_source_id = source_id
            if candidate.source_ids:
                bound_source_id = candidate.source_ids[0]
                if source_id is not None and source_id != bound_source_id:
                    raise ValueError(
                        "learning candidate acceptance must use its bound source"
                    )
                effective_source_id = bound_source_id
            candidate_text = await self._candidate_reviewer.acceptance_context(
                self.identity.id,
                candidate.id,
                effective_source_id,
            )
            loop = self._require_loop()
            run_id = self._id_factory("run")
            result: LoopExit | None = None
            run_error: BaseException | None = None
            try:
                result = await self._run_locked(
                    loop,
                    (
                        "Review the explicitly selected inactive learning candidate "
                        "against current catalog and active knowledge. If it remains "
                        "correct, durable, grounded, scoped, and non-duplicate, "
                        "propose the exact matching existing mutation before "
                        "returning text. Otherwise do not mutate active knowledge."
                    ),
                    conversation_id=conversation_id,
                    source_id=effective_source_id,
                    learning_candidate_id=candidate.id,
                    learning_candidate_text=candidate_text,
                    learning_candidate=candidate,
                    run_id=run_id,
                )
            except BaseException as error:
                run_error = error

            finalization_cancelled = False
            try:
                if self._learning_candidate_guard.mutation_succeeded(run_id):
                    _, finalization_cancelled = await _await_async_completion(
                        lambda: self._candidate_reviewer.mark_accepted(
                            candidate.id,
                            expected_fingerprint=candidate.candidate_fingerprint,
                        )
                    )
            finally:
                self._learning_candidate_guard.clear_outcome(run_id)

            if run_error is not None:
                raise run_error.with_traceback(run_error.__traceback__)
            if finalization_cancelled:
                raise asyncio.CancelledError
            assert result is not None
            return result

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
        async with self._mutation_lock:
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
        async with self._mutation_lock:
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
    ) -> bool:
        self._require_open()
        return await self._skill_store.save_skill(name, description, instructions)

    async def delete_skill(self, name: str) -> bool:
        self._require_open()
        return await self._skill_store.delete_skill(name)

    async def inspect_mcp_server(
        self,
        *,
        endpoint: str,
        authentication: MCPAuthentication | None = None,
    ) -> MCPServerInspection:
        """Inspect one exact endpoint without persisting execution authority."""

        async with self._mutation_lock:
            self._require_open()
            return await self._inspect_mcp_endpoint(
                endpoint=endpoint,
                authentication=authentication or MCPAuthentication.no_auth(),
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
    ) -> MCPBindingStatus:
        """Persist one exact binding; declarations activate only after reopen."""

        if not isinstance(maximum_outbound_sensitivity, ModelSensitivity):
            raise TypeError("maximum_outbound_sensitivity is invalid")
        resolved_authentication = authentication or MCPAuthentication.no_auth()
        resolved_binding_id = binding_id or self._id_factory("mcp-binding")
        async with self._run_lock:
            async with self._mutation_lock:
                self._require_open()
                current = await self._store.load_mcp_binding(
                    self.identity.id,
                    resolved_binding_id,
                )
                inspection = await self._inspect_mcp_endpoint(
                    endpoint=endpoint,
                    authentication=resolved_authentication,
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
                )
                stored = await self._store.store_mcp_binding(
                    binding,
                    expected_revision=(None if current is None else current.revision),
                )
                await self._deactivate_mcp_binding(stored.binding_id)
                return MCPBindingStatus(stored, None)

    async def list_mcp_servers(self) -> tuple[MCPBindingStatus, ...]:
        """Return bounded non-secret status for all independently keyed bindings."""

        self._require_open()
        bindings = await self._store.list_mcp_bindings(self.identity.id)
        statuses: list[MCPBindingStatus] = []
        for binding in bindings:
            activated = self._mcp_activated_bindings.get(binding.binding_id)
            statuses.append(
                MCPBindingStatus(
                    binding,
                    None if activated is None else activated.binding.revision,
                )
            )
        return tuple(statuses)

    async def refresh_mcp_server(self, binding_id: str) -> MCPBindingStatus:
        """Check exact remote drift and require reopen for any refreshed revision."""

        async with self._run_lock:
            async with self._mutation_lock:
                self._require_open()
                current = await self._store.load_mcp_binding(
                    self.identity.id,
                    binding_id,
                )
                if current is None:
                    raise ValueError("MCP binding does not exist")
                if current.state is MCPBindingState.REVOKED:
                    raise ValueError(
                        "revoked MCP binding must be explicitly reattached"
                    )
                inspection = await self._inspect_mcp_endpoint(
                    endpoint=current.endpoint,
                    authentication=current.authentication,
                )
                refreshed = current.checked(
                    observed_at=inspection.observed_at,
                    stale_reason=mcp_binding_drift_reason(current, inspection),
                )
                stored = await self._store.store_mcp_binding(
                    refreshed,
                    expected_revision=current.revision,
                )
                await self._deactivate_mcp_binding(stored.binding_id)
                return MCPBindingStatus(stored, None)

    async def revoke_mcp_server(self, binding_id: str) -> MCPBindingStatus:
        """Make one binding immediately unavailable without affecting siblings."""

        async with self._mutation_lock:
            self._require_open()
            activated = self._mcp_activated_bindings.get(binding_id)
            if activated is None:
                current = await self._store.load_mcp_binding(
                    self.identity.id,
                    binding_id,
                )
                if current is None:
                    raise ValueError("MCP binding does not exist")
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
    ) -> MCPServerInspection:
        client = self._mcp_client_factory.create(
            endpoint=endpoint,
            authentication=authentication,
            secrets=self._secret_provider,
        )
        try:
            return await client.inspect(observed_at=self._clock())
        finally:
            await client.close()

    async def _deactivate_mcp_binding(self, binding_id: str) -> None:
        activated = self._mcp_activated_bindings.pop(binding_id, None)
        if activated is not None:
            await activated.executor.close()

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        return await self._attach_source(source, attached_at=self._clock())

    async def _attach_source(
        self,
        source: ResourceSource,
        *,
        attached_at: datetime,
    ) -> SourceRegistration:
        async with self._run_lock:
            async with self._mutation_lock:
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
            if persisted_registration is not None:
                if persisted_registration != opened_registration:
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
                )
            )
            await self._store.commit_snapshot(
                discovery.snapshot,
                registration=registration,
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
        async with self._run_lock:
            async with self._mutation_lock:
                self._require_open()
                current = await self._store.load_source(self.identity.id, source_id)
                if current is None or not current.active:
                    raise ValueError("source edit requires an active owned source")
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
                    old_reference = _owned_source_credential_reference(
                        current,
                        agent_id=self.identity.id,
                    )
                    new_reference = _owned_source_credential_reference(
                        registration,
                        agent_id=self.identity.id,
                    )
                    previous_credential_deleted = True
                    if old_reference is not None and old_reference != new_reference:
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

    async def attach_local_directory(
        self,
        root: str | Path,
        *,
        name: str | None = None,
    ) -> SourceRegistration:
        """Attach one ordinary bounded CSV/JSON directory source."""

        return await self.attach(LocalDirectorySource(root=root, name=name))

    async def edit_local_directory_source(
        self,
        source_id: str,
        root: str | Path,
        *,
        confirmation_handler: SourceEditConfirmationHandler,
        name: str | None = None,
        max_depth: int = 8,
        max_files: int = 1_000,
        max_file_bytes: int = 2 * 1024 * 1024,
        max_columns: int = 512,
        max_rows: int = 100_000,
        max_json_nodes: int = 500_000,
        max_json_depth: int = 32,
        max_key_bytes: int = 1_024,
        max_string_bytes: int = 256 * 1024,
        max_cell_bytes: int = 1024 * 1024,
    ) -> SourceEditResult | None:
        return await self.edit_source(
            source_id,
            LocalDirectorySource(
                root=root,
                name=name,
                max_depth=max_depth,
                max_files=max_files,
                max_file_bytes=max_file_bytes,
                max_columns=max_columns,
                max_rows=max_rows,
                max_json_nodes=max_json_nodes,
                max_json_depth=max_json_depth,
                max_key_bytes=max_key_bytes,
                max_string_bytes=max_string_bytes,
                max_cell_bytes=max_cell_bytes,
            ),
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
        async with self._mutation_lock:
            self._require_open()
            reference = SecretReference.keychain(
                _credential_account(
                    self.identity.id,
                    "postgresql",
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
                _credential_account_prefix(self.identity.id, "postgresql")
            )
        ):
            raise ValueError(
                "credential does not belong to this agent's PostgreSQL setup"
            )
        async with self._mutation_lock:
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

        async with self._mutation_lock:
            self._require_open()
            return await self._inspect_source_permissions_locked(source_id)

    async def preview_source_permissions(
        self,
        *,
        source_id: str,
        read_mode: SourceReadMode,
        read_resource_ids: tuple[str, ...],
        postgresql_update_scopes: Mapping[str, tuple[str, ...]],
    ) -> SourcePermissionsPreview:
        """Build and retain one bounded in-process exact confirmation preview."""

        async with self._run_lock:
            async with self._mutation_lock:
                self._require_open()
                inspection = await self._inspect_source_permissions_locked(source_id)
                preview = await self._build_source_permissions_preview(
                    inspection,
                    read_mode=read_mode,
                    read_resource_ids=read_resource_ids,
                    postgresql_update_scopes=postgresql_update_scopes,
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

        async with self._run_lock:
            async with self._mutation_lock:
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
                    scope.resource_id: scope.allowed_assignment_columns
                    for scope in preview.after.postgresql_update_scopes
                }
                revalidated = await self._build_source_permissions_preview(
                    current,
                    read_mode=preview.after.read_scope.mode,
                    read_resource_ids=preview.after.read_scope.resource_ids,
                    postgresql_update_scopes=update_mapping,
                )
                if revalidated.after != preview.after:
                    raise ValueError(
                        "source permission facts changed after preview; preview again"
                    )
                await self._store.replace_source_permission_scopes(
                    preview.after.read_scope,
                    preview.after.postgresql_update_scopes,
                )
                return await self._inspect_source_permissions_locked(source_id)

    async def _inspect_source_permissions_locked(
        self,
        source_id: str,
    ) -> SourcePermissionsInspection:
        registration = await self._store.load_source(self.identity.id, source_id)
        if (
            registration is None
            or registration.agent_id != self.identity.id
            or registration.id != source_id
            or not registration.active
        ):
            raise ValueError("unknown active source for this agent")
        read_scope = await self._store.load_source_read_scope(
            self.identity.id,
            source_id,
        )
        if read_scope is None:
            raise ValueError("active source is missing its read scope")
        update_scopes = await self._store.list_postgresql_update_scopes(
            self.identity.id,
            source_id,
        )
        refs = await self._store.list_current_snapshot_refs(
            self.identity.id,
            (source_id,),
        )
        if len(refs) > 1:
            raise ValueError("source has multiple current catalog generations")
        catalog_generation = None if not refs else refs[0].sync_id
        resources = tuple(
            resource
            for resource in await self._store.list_resources(
                self.identity.id,
                source_id,
            )
            if resource.agent_id == self.identity.id and resource.source_id == source_id
        )
        schemas = (
            await self._data_view.resource_schemas(self.identity.id, source_id)
            if registration.adapter_id == "postgresql"
            else ()
        )
        schemas_by_resource_id = {schema.resource_id: schema for schema in schemas}
        choices: list[SourcePermissionResource] = []
        for resource in resources:
            eligible_columns: tuple[str, ...] = ()
            schema = schemas_by_resource_id.get(resource.id)
            if schema is not None and resource.kind is ResourceKind.TABLE:
                eligible_columns = tuple(
                    column
                    for column in schema.columns
                    if validate_postgresql_update_scope(
                        source_id,
                        resource.id,
                        (column,),
                        resources=(schema,),
                    ).valid
                )
            choices.append(
                SourcePermissionResource(
                    resource_id=resource.id,
                    display_name=resource.native_identity,
                    resource_kind=resource.kind.value,
                    eligible_assignment_columns=eligible_columns,
                )
            )
        return SourcePermissionsInspection(
            source_id=source_id,
            source_display_name=registration.display_name,
            adapter_id=registration.adapter_id,
            catalog_generation=catalog_generation,
            state=SourcePermissionState(read_scope, update_scopes),
            resources=tuple(choices),
        )

    async def _build_source_permissions_preview(
        self,
        inspection: SourcePermissionsInspection,
        *,
        read_mode: SourceReadMode,
        read_resource_ids: tuple[str, ...],
        postgresql_update_scopes: Mapping[str, tuple[str, ...]],
    ) -> SourcePermissionsPreview:
        if not isinstance(read_mode, SourceReadMode):
            raise TypeError("read_mode must be SourceReadMode")
        if not isinstance(read_resource_ids, tuple):
            raise TypeError("read_resource_ids must be a tuple")
        if not isinstance(postgresql_update_scopes, Mapping):
            raise TypeError("postgresql_update_scopes must be a mapping")
        current_resources = {
            resource.resource_id: resource for resource in inspection.resources
        }
        requested_read = SourceReadScope(
            agent_id=self.identity.id,
            source_id=inspection.source_id,
            mode=read_mode,
            resource_ids=read_resource_ids,
        )
        if requested_read.mode is SourceReadMode.SELECTED:
            unknown_read_ids = set(requested_read.resource_ids) - set(current_resources)
            if unknown_read_ids:
                raise ValueError(
                    "selected read resources are not current resources of this source"
                )
        if postgresql_update_scopes and inspection.adapter_id != "postgresql":
            raise ValueError(
                "PostgreSQL update scopes require an active PostgreSQL source"
            )

        registration = await self._store.load_source(
            self.identity.id,
            inspection.source_id,
        )
        if (
            registration is None
            or not registration.active
            or registration.adapter_id != inspection.adapter_id
        ):
            raise ValueError("source identity changed during permission preview")
        tabular_by_resource_id = {
            item.resource.id: item
            for item in await self._catalog_service.tabular_resources(
                self.identity.id,
                inspection.source_id,
            )
        }
        update_scopes: list[PostgreSQLUpdateScope] = []
        requested_update_ids: set[str] = set()
        for resource_id, raw_columns in postgresql_update_scopes.items():
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError("update scope resource ids must be non-empty text")
            if not isinstance(raw_columns, tuple):
                raise TypeError("update scope columns must be tuples")
            resource = current_resources.get(resource_id)
            if resource is None or not resource.postgresql_update_eligible:
                raise ValueError(
                    "PostgreSQL update scope requires a current eligible table"
                )
            selected_columns = tuple(sorted(raw_columns))
            if (
                not selected_columns
                or len(selected_columns) != len(set(selected_columns))
                or not set(selected_columns)
                <= set(resource.eligible_assignment_columns)
            ):
                raise ValueError(
                    "update scope columns must be a non-empty eligible subset"
                )
            requested_update_ids.add(resource_id)
            tabular = tabular_by_resource_id.get(resource_id)
            if tabular is None or tabular.resource.kind is not ResourceKind.TABLE:
                raise ValueError(
                    "PostgreSQL update scope requires exact current table facts"
                )
            update_scopes.append(
                PostgreSQLUpdateScope(
                    agent_id=self.identity.id,
                    source_id=inspection.source_id,
                    resource_id=resource_id,
                    allowed_assignment_columns=selected_columns,
                    authorization_fingerprint=(
                        postgresql_update_authorization_fingerprint(
                            source=registration,
                            resource=tabular.resource,
                            facet=tabular.facet,
                            allowed_assignment_columns=selected_columns,
                        )
                    ),
                )
            )

        automatic_read_additions: set[str] = set()
        if requested_read.mode is SourceReadMode.ALL:
            final_read_scope = requested_read
        else:
            requested_ids = set(requested_read.resource_ids)
            automatic_read_additions = requested_update_ids - requested_ids
            final_ids = requested_ids | requested_update_ids
            final_read_scope = (
                SourceReadScope(
                    agent_id=self.identity.id,
                    source_id=inspection.source_id,
                    mode=SourceReadMode.SELECTED,
                    resource_ids=tuple(final_ids),
                )
                if final_ids
                else SourceReadScope(
                    agent_id=self.identity.id,
                    source_id=inspection.source_id,
                    mode=SourceReadMode.NONE,
                )
            )
        effective_read_ids = (
            set(current_resources)
            if final_read_scope.mode is SourceReadMode.ALL
            else set(final_read_scope.resource_ids)
        )
        dependent_revocations = tuple(
            scope.resource_id
            for scope in inspection.state.postgresql_update_scopes
            if scope.resource_id not in effective_read_ids
        )

        after = SourcePermissionState(
            final_read_scope,
            tuple(update_scopes),
        )
        names = {
            resource.resource_id: resource.display_name
            for resource in inspection.resources
        }
        summary = SourcePermissionSummary(
            source_display_name=inspection.source_display_name,
            read_mode=final_read_scope.mode,
            selected_read_resource_count=(
                len(current_resources)
                if final_read_scope.mode is SourceReadMode.ALL
                else len(final_read_scope.resource_ids)
            ),
            postgresql_update_table_count=len(update_scopes),
            postgresql_update_table_examples=tuple(
                names.get(scope.resource_id, scope.resource_id)
                for scope in sorted(
                    update_scopes,
                    key=lambda item: names.get(item.resource_id, item.resource_id),
                )[:5]
            ),
            automatic_read_addition_examples=tuple(
                names.get(resource_id, resource_id)
                for resource_id in sorted(
                    automatic_read_additions,
                    key=lambda item: names.get(item, item),
                )[:5]
            ),
            dependent_update_revocation_examples=tuple(
                names.get(resource_id, resource_id)
                for resource_id in sorted(
                    dependent_revocations,
                    key=lambda item: names.get(item, item),
                )[:5]
            ),
        )
        material = {
            "source_id": inspection.source_id,
            "adapter_id": inspection.adapter_id,
            "catalog_generation": inspection.catalog_generation,
            "before": _source_permission_state_payload(inspection.state),
            "after": _source_permission_state_payload(after),
            "automatic_read_additions": tuple(sorted(automatic_read_additions)),
            "dependent_update_revocations": tuple(sorted(dependent_revocations)),
        }
        confirmation_fingerprint = (
            "sha256:"
            + hmac.new(
                self._source_permission_confirmation_key,
                canonical_json(material).encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
        )
        return SourcePermissionsPreview(
            source_id=inspection.source_id,
            catalog_generation=inspection.catalog_generation,
            before=inspection.state,
            after=after,
            automatic_read_additions=tuple(automatic_read_additions),
            dependent_update_revocations=dependent_revocations,
            summary=summary,
            confirmation_fingerprint=confirmation_fingerprint,
        )

    async def postgresql_update_readiness(
        self,
        source_id: str,
        resource_id: str,
        assignment_columns: tuple[str, ...],
    ) -> PostgreSQLUpdateReadiness:
        """Inspect one exact PostgreSQL update scope without mutating state."""

        self._require_open()
        return await self._postgresql_update_backend.postgresql_update_readiness(
            agent_id=self.identity.id,
            source_id=source_id,
            resource_id=resource_id,
            assignment_columns=assignment_columns,
        )

    async def detach(self, source_id: str) -> SourceRegistration:
        async with self._run_lock:
            async with self._mutation_lock:
                self._require_open()
                detached = await self._store.detach_source(
                    self.identity.id, source_id, self._clock()
                )
                reference = _owned_source_credential_reference(
                    detached,
                    agent_id=self.identity.id,
                )
                if reference is not None:
                    try:
                        await self._keychain.delete(reference)
                    except Exception:
                        raise AgentHomeError(
                            "source was detached, but its stored credential "
                            "could not be deleted"
                        ) from None
                return detached

    async def refresh_source(self, source_id: str) -> SourceRegistration:
        """Refresh one active source using its exact admitted registration."""

        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")
        async with self._run_lock:
            async with self._mutation_lock:
                self._require_open()
                registration = await self._store.load_source(
                    self.identity.id, source_id
                )
                if registration is None or not registration.active:
                    raise ValueError("unknown active source for this agent")
                source = _source_from_registration(
                    registration,
                    secret_provider=default_secret_provider(
                        self._secret_provider or self._keychain
                    ),
                )
                return await self._attach_source_locked(
                    source,
                    attached_at=registration.attached_at,
                    persisted_registration=registration,
                )

    async def list_sources(self) -> tuple[SourceRegistration, ...]:
        self._require_open()
        return await self._store.list_sources(self.identity.id)

    async def list_catalog_resources(
        self, *, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        self._require_open()
        return await self._store.list_resources(self.identity.id, source_id)

    async def catalog_summary(self) -> CatalogSummary:
        """Return current committed catalog facts as one consistent projection."""

        async with self._mutation_lock:
            self._require_open()
            return await self._catalog_service.summary(self.identity.id)

    async def catalog_preview(
        self,
        *,
        limit: int = 12,
    ) -> tuple[CatalogResource, ...]:
        """Return a bounded deterministic preview from current catalog truth."""

        async with self._mutation_lock:
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
        followup_driver = self._followup_driver
        self._followup_driver = None
        if followup_driver is not None:
            self._followup_wake.set()
            followup_driver.cancel("host_closing")
            try:
                await followup_driver
            except asyncio.CancelledError:
                pass
            except BaseException as error:
                first_error = error
        try:
            await self._job_supervisor.close()
        except BaseException as error:
            first_error = error
        async with self._run_lock:
            async with self._mutation_lock:
                pass
        activated_bindings = tuple(self._mcp_activated_bindings.values())
        self._mcp_activated_bindings.clear()
        for activated in activated_bindings:
            try:
                await activated.executor.close()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        for store in (
            self._candidate_reviewer,
            self._memory_store,
            self._skill_store,
            self._artifact_delivery,
            self._artifact_store,
            self._store,
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
        self._writer_lock.release()
        if first_error is not None:
            raise first_error

    def _require_loop(self) -> AgentLoop:
        self._require_open()
        if self._loop is None:
            raise AgentNotConfiguredError("agent execution requires a model")
        return self._loop

    def _require_open(self) -> None:
        if self._closed:
            raise AgentHomeError("embedded agent is closed")


def _source_permission_state_payload(
    state: SourcePermissionState,
) -> dict[str, object]:
    return {
        "read": {
            "mode": state.read_scope.mode.value,
            "resource_ids": state.read_scope.resource_ids,
        },
        "postgresql_updates": tuple(
            {
                "resource_id": scope.resource_id,
                "allowed_assignment_columns": scope.allowed_assignment_columns,
                "authorization_fingerprint": scope.authorization_fingerprint,
            }
            for scope in state.postgresql_update_scopes
        ),
    }


def _source_from_registration(
    registration: SourceRegistration,
    *,
    secret_provider: SecretProvider,
) -> ResourceSource:
    """Reconstruct one admitted source only at the composition boundary."""

    configuration = registration.configuration
    adapter_id = registration.adapter_id
    if adapter_id == "sqlite":
        _require_configuration_fields(configuration, {"path"})
        return SQLiteSource(
            path=_configuration_text(configuration, "path"),
            name=registration.display_name,
        )
    if adapter_id == "local-directory":
        fields = {
            "formats",
            "max_cell_bytes",
            "max_columns",
            "max_depth",
            "max_file_bytes",
            "max_files",
            "max_json_depth",
            "max_json_nodes",
            "max_key_bytes",
            "max_rows",
            "max_string_bytes",
            "root",
            "root_device",
            "root_inode",
        }
        _require_configuration_fields(configuration, fields)
        if configuration["formats"] != ("csv", "json"):
            raise AgentHomeError("local-directory source configuration is invalid")
        _configuration_integer(configuration, "root_device")
        _configuration_integer(configuration, "root_inode")
        return LocalDirectorySource(
            root=_configuration_text(configuration, "root"),
            name=registration.display_name,
            max_depth=_configuration_integer(configuration, "max_depth"),
            max_files=_configuration_integer(configuration, "max_files"),
            max_file_bytes=_configuration_integer(
                configuration,
                "max_file_bytes",
            ),
            max_columns=_configuration_integer(configuration, "max_columns"),
            max_rows=_configuration_integer(configuration, "max_rows"),
            max_json_nodes=_configuration_integer(
                configuration,
                "max_json_nodes",
            ),
            max_json_depth=_configuration_integer(
                configuration,
                "max_json_depth",
            ),
            max_key_bytes=_configuration_integer(
                configuration,
                "max_key_bytes",
            ),
            max_string_bytes=_configuration_integer(
                configuration,
                "max_string_bytes",
            ),
            max_cell_bytes=_configuration_integer(
                configuration,
                "max_cell_bytes",
            ),
        )
    if adapter_id == "postgresql":
        required = {
            "database",
            "host",
            "port",
            "schemas",
            "ssl_mode",
            "username",
        }
        allowed = required | {"credential_ref"}
        fields = set(configuration)
        if not required <= fields or not fields <= allowed:
            raise AgentHomeError("PostgreSQL source configuration is invalid")
        raw_schemas = configuration["schemas"]
        if not isinstance(raw_schemas, tuple) or any(
            not isinstance(schema, str) for schema in raw_schemas
        ):
            raise AgentHomeError("PostgreSQL source configuration is invalid")
        raw_reference = configuration.get("credential_ref")
        if raw_reference is not None and not isinstance(raw_reference, str):
            raise AgentHomeError("PostgreSQL source configuration is invalid")
        try:
            reference = (
                None if raw_reference is None else SecretReference.parse(raw_reference)
            )
            return PostgreSQLSource(
                host=_configuration_text(configuration, "host"),
                port=_configuration_integer(configuration, "port"),
                database=_configuration_text(configuration, "database"),
                username=_configuration_text(configuration, "username"),
                credential=reference,
                schemas=cast(tuple[str, ...], raw_schemas),
                ssl_mode=_configuration_text(configuration, "ssl_mode"),
                name=registration.display_name,
                secret_provider=secret_provider,
            )
        except (TypeError, ValueError) as error:
            raise AgentHomeError(
                "PostgreSQL source configuration is invalid"
            ) from error
    raise AgentHomeError("registered source adapter cannot be refreshed")


def _resolve_candidate_reviewer_profile(
    model: ModelProvider | None,
    profile: ModelProfile | None,
) -> ModelProfile | None:
    """Admit one direct provider profile without retry or fallback."""

    if model is None:
        if profile is not None:
            raise ValueError("reviewer_profile requires reviewer_model")
        return None
    if isinstance(model, ModelRouter):
        raise ValueError(
            "candidate reviewer requires one direct provider without fallback"
        )
    resolved = profile
    if resolved is None:
        candidate = getattr(model, "model_profile", None)
        if isinstance(candidate, ModelProfile):
            resolved = candidate
    if not isinstance(resolved, ModelProfile):
        raise ValueError("candidate reviewer requires an explicit bounded ModelProfile")
    if resolved.max_output_tokens >= LEARNING_REVIEW_MAX_TOTAL_TOKENS:
        raise ValueError(
            "candidate reviewer profile output limit exceeds review token budget"
        )
    return resolved


def _candidate_reviewer_from_route(
    route: ModelRoute,
    *,
    secret_provider: SecretProvider | None,
) -> tuple[ModelProvider, ModelProfile]:
    """Derive one bounded, direct reviewer from the persisted primary route."""

    primary = route.candidates[0]
    profile = replace(
        primary.profile,
        max_output_tokens=min(
            primary.profile.max_output_tokens,
            _CANDIDATE_REVIEWER_MAX_OUTPUT_TOKENS,
        ),
    )
    direct_route = ModelRoute(
        (
            ModelRouteCandidate(
                provider_id=primary.provider_id,
                profile=profile,
                base_url=primary.base_url,
                secret_reference=primary.secret_reference,
                allowed_sensitivities=primary.allowed_sensitivities,
            ),
        ),
        retry_policy=RetryPolicy(attempts=1, backoff_seconds=0),
    )
    return (
        create_model_route_provider(
            direct_route,
            secret_provider=secret_provider,
        ),
        profile,
    )


def _require_configuration_fields(
    configuration: Mapping[str, object],
    expected: set[str],
) -> None:
    if set(configuration) != expected:
        raise AgentHomeError("source configuration fields are invalid")


def _configuration_text(
    configuration: Mapping[str, object],
    name: str,
) -> str:
    value = configuration[name]
    if not isinstance(value, str) or not value:
        raise AgentHomeError("source configuration text is invalid")
    return value


def _configuration_integer(
    configuration: Mapping[str, object],
    name: str,
) -> int:
    value = configuration[name]
    if not isinstance(value, int) or isinstance(value, bool):
        raise AgentHomeError("source configuration integer is invalid")
    return value


def _resolve_configuration(
    config: AgentConfig | None,
    *,
    model: ModelProvider | None,
    model_profile: ModelProfile | None,
    limits: LoopLimits | None,
    secret_provider: SecretProvider | None,
) -> tuple[ModelProvider | None, ModelProfile | None, ModelRoute | None, LoopLimits]:
    if config is not None and not isinstance(config, AgentConfig):
        raise TypeError("config must be AgentConfig or None")
    route = None if config is None else config.model_route
    if route is not None:
        if model is not None:
            raise AgentNotConfiguredError(
                "model_route and injected model cannot be combined"
            )
        model = create_model_route_provider(route, secret_provider=secret_provider)
        model_profile = route.model_profile
    resolved_limits = limits or (LoopLimits() if config is None else config.limits)
    if model is not None and model_profile is None:
        raise AgentNotConfiguredError("a configured model requires its ModelProfile")
    if model is not None:
        assert model_profile is not None
        if model.provider_id != model_profile.id:
            raise AgentNotConfiguredError("model and profile identities differ")
    return model, model_profile, route, resolved_limits


def _resolve_credential_session(
    keychain: KeychainStore | None,
) -> tuple[CredentialSession, bool]:
    if isinstance(keychain, CredentialSession):
        return keychain, False
    return CredentialSession(keychain or KeychainSecretProvider()), True


def _configuration_was_injected(
    config: AgentConfig | None,
    *,
    model: ModelProvider | None,
    model_profile: ModelProfile | None,
    context_builder: ContextBuilder | None,
    tools: ToolRuntime | None,
) -> bool:
    return any(
        value is not None
        for value in (
            config,
            model,
            model_profile,
            context_builder,
            tools,
        )
    )


def _admit_model_selection(
    provider: str,
    model: str,
    base_url: str | None,
) -> tuple[str, str, str | None, bool]:
    if not isinstance(provider, str):
        raise TypeError("provider must be a string")
    provider_name = provider.strip().lower()
    if _PROVIDER_NAME.fullmatch(provider_name) is None:
        raise ValueError("provider must be a bounded lowercase identifier")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model identifier must be non-empty")
    model_name = model.strip()
    if any(
        character.isspace() or ord(character) < 32 or ord(character) == 127
        for character in model_name
    ):
        raise ValueError("model identifier cannot contain whitespace or controls")
    endpoint: str | None = None
    if base_url is not None:
        if not isinstance(base_url, str) or not base_url.strip():
            raise ValueError("base URL must be non-empty when provided")
        endpoint = base_url.strip()
        if (
            len(endpoint) > 2_048
            or any(
                ord(character) < 32 or ord(character) == 127 for character in endpoint
            )
            or not endpoint.startswith(("http://", "https://"))
        ):
            raise ValueError("base URL must be a bounded HTTP or HTTPS URL")
    if provider_name in _BUILTIN_PROVIDERS - {"ollama"} and endpoint is not None:
        raise ValueError(f"{provider_name} uses its fixed endpoint")
    if provider_name not in _BUILTIN_PROVIDERS and endpoint is None:
        raise ValueError("custom providers require an explicit base URL")
    provider_id = f"{provider_name}:{model_name}"
    if len(provider_id) > 256:
        raise ValueError("model identity exceeds its 256 character bound")
    return (
        provider_name,
        model_name,
        endpoint,
        provider_name not in {"ollama", "claude-code", "grok-build"},
    )


def _model_profile(
    provider: str,
    model: str,
    *,
    context_window_tokens: int | None,
    max_output_tokens: int | None,
) -> ModelProfile:
    provider_id = f"{provider}:{model}"
    reviewed = reviewed_model_profile(provider_id)
    if reviewed is not None:
        if context_window_tokens is not None or max_output_tokens is not None:
            raise ValueError("reviewed models use their canonical profile limits")
        return reviewed
    if context_window_tokens is None or max_output_tokens is None:
        raise ValueError(
            "unreviewed models require explicit context and output token limits"
        )
    return ModelProfile(
        id=provider_id,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
        supports_tools=True,
        supports_parallel_tools=provider in _SUBSCRIPTION_PROVIDERS,
        supports_structured_output=provider in _SUBSCRIPTION_PROVIDERS,
        supports_streaming=(
            provider in _BUILTIN_PROVIDERS and provider not in _SUBSCRIPTION_PROVIDERS
        ),
        supports_reasoning=provider in _SUBSCRIPTION_PROVIDERS,
    )


def _model_route(
    provider: str,
    model: str,
    *,
    base_url: str | None,
    secret_reference: SecretReference | None,
    context_window_tokens: int | None,
    max_output_tokens: int | None,
) -> ModelRoute:
    profile = _model_profile(
        provider,
        model,
        context_window_tokens=context_window_tokens,
        max_output_tokens=max_output_tokens,
    )
    return ModelRoute(
        (
            ModelRouteCandidate(
                provider_id=profile.id,
                profile=profile,
                base_url=base_url,
                secret_reference=secret_reference,
            ),
        )
    )


def _credential_account(agent_id: str, provider: str, nonce: str) -> str:
    digest = hashlib.sha256(
        f"{agent_id}\x00{provider}\x00{nonce}".encode("utf-8")
    ).hexdigest()[:24]
    return _credential_account_prefix(agent_id, provider) + digest


def _credential_account_prefix(agent_id: str, provider: str) -> str:
    prefix = f"{agent_id}:{provider}:"
    if len(prefix) + 24 > 256:
        agent_digest = hashlib.sha256(agent_id.encode("utf-8")).hexdigest()[:24]
        prefix = f"agent-{agent_digest}:{provider}:"
    return prefix


def _credential_reference_is_owned(
    reference: SecretReference,
    *,
    agent_id: str,
    provider: str,
) -> bool:
    if reference.scheme != "keychain":
        return False
    prefix = _credential_account_prefix(agent_id, provider)
    if not reference.name.startswith(prefix):
        return False
    suffix = reference.name[len(prefix) :]
    return len(suffix) == 24 and all(
        character in "0123456789abcdef" for character in suffix
    )


def _source_edit_read_scope(
    *,
    current_read_scope: SourceReadScope,
    current_resources: tuple[CatalogResource, ...],
    replacement: SourceCatalogSnapshot,
) -> tuple[SourceReadScope, tuple[str, ...]]:
    """Carry read intent only across exact kind/native-identity matches."""

    if current_read_scope.mode is SourceReadMode.ALL:
        return (
            SourceReadScope.allow_all(
                agent_id=replacement.sync.agent_id,
                source_id=replacement.sync.source_id,
            ),
            (),
        )
    if current_read_scope.mode is SourceReadMode.NONE:
        return (
            SourceReadScope(
                agent_id=replacement.sync.agent_id,
                source_id=replacement.sync.source_id,
                mode=SourceReadMode.NONE,
            ),
            (),
        )

    current_by_id = {resource.id: resource for resource in current_resources}
    replacement_by_identity: dict[tuple[ResourceKind, str], list[CatalogResource]] = {}
    for resource in replacement.resources:
        replacement_by_identity.setdefault(
            (resource.kind, resource.native_identity),
            [],
        ).append(resource)
    preserved: list[str] = []
    omitted: list[str] = []
    for resource_id in current_read_scope.resource_ids:
        current = current_by_id.get(resource_id)
        if current is None:
            omitted.append(resource_id)
            continue
        matches = replacement_by_identity.get(
            (current.kind, current.native_identity),
            [],
        )
        if len(matches) != 1:
            omitted.append(current.native_identity)
            continue
        preserved.append(matches[0].id)
    if not preserved:
        return (
            SourceReadScope(
                agent_id=replacement.sync.agent_id,
                source_id=replacement.sync.source_id,
                mode=SourceReadMode.NONE,
            ),
            tuple(sorted(omitted)),
        )
    return (
        SourceReadScope(
            agent_id=replacement.sync.agent_id,
            source_id=replacement.sync.source_id,
            mode=SourceReadMode.SELECTED,
            resource_ids=tuple(preserved),
        ),
        tuple(sorted(omitted)),
    )


def _owned_source_credential_reference(
    registration: SourceRegistration,
    *,
    agent_id: str,
) -> SecretReference | None:
    if registration.agent_id != agent_id or registration.adapter_id != "postgresql":
        return None
    value = registration.configuration.get("credential_ref")
    if not isinstance(value, str):
        return None
    try:
        reference = SecretReference.parse(value)
    except ValueError:
        return None
    return (
        reference
        if _credential_reference_is_owned(
            reference,
            agent_id=agent_id,
            provider="postgresql",
        )
        else None
    )


def _owned_agent_credential_references(
    agent_id: str,
    *,
    model_document: object | None,
    sources: tuple[SourceRegistration, ...],
) -> tuple[SecretReference, ...]:
    references: dict[str, SecretReference] = {}
    if model_document is not None:
        if not isinstance(model_document, dict):
            raise AgentHomeError(
                "model configuration is invalid; the agent home was preserved"
            )
        route = model_document.get("model_route")
        candidates = route.get("candidates") if isinstance(route, dict) else None
        if not isinstance(candidates, list):
            raise AgentHomeError(
                "model configuration is invalid; the agent home was preserved"
            )
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise AgentHomeError(
                    "model configuration is invalid; the agent home was preserved"
                )
            provider_id = candidate.get("provider_id")
            reference_value = candidate.get("secret_reference")
            if not isinstance(provider_id, str) or (
                reference_value is not None and not isinstance(reference_value, str)
            ):
                raise AgentHomeError(
                    "model configuration is invalid; the agent home was preserved"
                )
            provider, separator, _model = provider_id.partition(":")
            if not separator or reference_value is None:
                continue
            try:
                reference = SecretReference.parse(reference_value)
            except ValueError:
                continue
            if _credential_reference_is_owned(
                reference,
                agent_id=agent_id,
                provider=provider,
            ):
                references[reference.to_uri()] = reference
    for source in sources:
        source_reference = _owned_source_credential_reference(
            source,
            agent_id=agent_id,
        )
        if source_reference is not None:
            references[source_reference.to_uri()] = source_reference
    return tuple(references[key] for key in sorted(references))


async def _validate_model_route(
    route: ModelRoute,
    *,
    secret_provider: SecretProvider,
    injected_provider: ModelProvider | None,
) -> None:
    validation_candidates = tuple(
        replace(
            candidate,
            profile=replace(
                candidate.profile,
                max_output_tokens=min(
                    (
                        _REASONING_MODEL_VALIDATION_MAX_OUTPUT_TOKENS
                        if candidate.profile.supports_reasoning
                        else _MODEL_VALIDATION_MAX_OUTPUT_TOKENS
                    ),
                    candidate.profile.max_output_tokens,
                    candidate.profile.context_window_tokens - 1,
                ),
            ),
        )
        for candidate in route.candidates
    )
    validation_route = ModelRoute(validation_candidates, route.retry_policy)
    if injected_provider is None:
        provider = create_model_route_provider(
            validation_route,
            secret_provider=secret_provider,
        )
    else:
        if len(validation_candidates) != 1:
            raise AgentNotConfiguredError(
                "an injected validation provider requires one route candidate"
            )
        candidate = validation_candidates[0]
        if injected_provider.provider_id != candidate.profile.id:
            raise AgentNotConfiguredError(
                "validation provider and configured model identities differ"
            )
        provider = ModelRouter(
            (
                ModelProviderRegistration(
                    provider=injected_provider,
                    profile=candidate.profile,
                    allowed_sensitivities=candidate.allowed_sensitivities,
                ),
            ),
            retry_policy=validation_route.retry_policy,
        )
    response = await provider.generate(
        ModelRequest(
            messages=(
                CanonicalMessage(
                    role=MessageRole.USER,
                    content=(TextBlock("Call the validation tool once."),),
                ),
            ),
            tools=(
                ToolDefinition(
                    name=_MODEL_VALIDATION_TOOL_NAME,
                    description="Prove provider tool-call compatibility.",
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                ),
            ),
            sensitivity=ModelSensitivity.PUBLIC,
            allow_parallel_tool_calls=False,
        )
    )
    if response.finish_reason is FinishReason.LENGTH:
        raise ModelProviderError(
            ProviderErrorCode.OUTPUT_LIMIT,
            "model exhausted the validation output budget",
        )
    if (
        response.finish_reason is not FinishReason.TOOL_CALLS
        or len(response.tool_calls) != 1
        or response.tool_calls[0].name != _MODEL_VALIDATION_TOOL_NAME
        or response.tool_calls[0].arguments
    ):
        raise ValueError("model validation did not prove native tool-call support")


def _keychain_references(
    config: AgentConfig | None,
) -> tuple[tuple[str, SecretReference], ...]:
    if config is None or config.model_route is None:
        return ()
    return tuple(
        (candidate.provider_id.partition(":")[0], reference)
        for candidate in config.model_route.candidates
        if (reference := candidate.secret_reference) is not None
        and reference.scheme == "keychain"
    )


def _read_model_configuration_document(home: Path) -> object | None:
    path = home / _MODEL_CONFIG_NAME
    if path.is_symlink():
        raise AgentHomeError("model configuration cannot be a symlink")
    if not path.exists():
        return None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise ValueError("configuration is not a regular file")
            chunks: list[bytes] = []
            size = 0
            while size <= _MAX_MODEL_CONFIG_BYTES:
                chunk = os.read(
                    descriptor, min(8_192, _MAX_MODEL_CONFIG_BYTES + 1 - size)
                )
                if not chunk:
                    break
                chunks.append(chunk)
                size += len(chunk)
            if size > _MAX_MODEL_CONFIG_BYTES:
                raise ValueError("configuration exceeds its byte bound")
        finally:
            os.close(descriptor)
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"invalid JSON constant: {value}")
            ),
        )
        return value
    except AgentHomeError:
        raise
    except Exception:
        raise AgentHomeError("model configuration is invalid") from None


def _read_model_configuration(home: Path, agent_id: str) -> AgentConfig | None:
    value = _read_model_configuration_document(home)
    if value is None:
        return None
    try:
        return _decode_agent_config(value, agent_id=agent_id)
    except AgentHomeError:
        raise
    except Exception:
        raise AgentHomeError("model configuration is invalid") from None


def _write_model_configuration(home: Path, config: AgentConfig) -> None:
    if not isinstance(config, AgentConfig) or config.model_route is None:
        raise ValueError("persisted model configuration requires a model route")
    path = home / _MODEL_CONFIG_NAME
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise AgentHomeError("model configuration path is invalid")
    data = json.dumps(
        _encode_agent_config(config),
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(data) > _MAX_MODEL_CONFIG_BYTES:
        raise AgentHomeError("model configuration exceeds its byte bound")
    temporary = home / f".{_MODEL_CONFIG_NAME}.{uuid4().hex}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
        directory = os.open(home, os.O_RDONLY)
        try:
            try:
                os.fsync(directory)
            except OSError:
                pass
        finally:
            os.close(directory)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _encode_agent_config(config: AgentConfig) -> dict[str, object]:
    route = config.model_route
    assert route is not None
    return {
        "limits": {
            "max_estimated_cost_usd": (
                None
                if config.limits.max_estimated_cost_usd is None
                else str(config.limits.max_estimated_cost_usd)
            ),
            "max_steps": config.limits.max_steps,
            "max_total_tokens": config.limits.max_total_tokens,
            "max_wall_time_seconds": config.limits.max_wall_time_seconds,
        },
        "model_route": {
            "candidates": [
                {
                    "allowed_sensitivities": sorted(
                        sensitivity.value
                        for sensitivity in candidate.allowed_sensitivities
                    ),
                    "base_url": candidate.base_url,
                    "profile": _encode_model_profile(candidate.profile),
                    "provider_id": candidate.provider_id,
                    "secret_reference": (
                        None
                        if candidate.secret_reference is None
                        else candidate.secret_reference.to_uri()
                    ),
                }
                for candidate in route.candidates
            ],
            "retry_policy": {
                "attempts": route.retry_policy.attempts,
                "backoff_seconds": route.retry_policy.backoff_seconds,
            },
        },
    }


def _encode_model_profile(profile: ModelProfile) -> dict[str, object]:
    return {
        "available": profile.available,
        "context_window_tokens": profile.context_window_tokens,
        "data_routing_classification": profile.data_routing_classification,
        "healthy": profile.healthy,
        "id": profile.id,
        "input_cost_per_million_usd": None,
        "max_output_tokens": profile.max_output_tokens,
        "output_cost_per_million_usd": None,
        "supports_documents": profile.supports_documents,
        "supports_native_continuation": profile.supports_native_continuation,
        "supports_parallel_tools": profile.supports_parallel_tools,
        "supports_prompt_caching": profile.supports_prompt_caching,
        "supports_reasoning": profile.supports_reasoning,
        "supports_streaming": profile.supports_streaming,
        "supports_structured_output": profile.supports_structured_output,
        "supports_tools": profile.supports_tools,
        "supports_vision": profile.supports_vision,
    }


def _decode_agent_config(value: object, *, agent_id: str) -> AgentConfig:
    document = _strict_mapping(value, {"limits", "model_route"})
    route_value = _strict_mapping(
        document["model_route"],
        {"candidates", "retry_policy"},
    )
    raw_candidates = route_value["candidates"]
    if not isinstance(raw_candidates, list) or not 1 <= len(raw_candidates) <= 5:
        raise ValueError("model route candidates are invalid")
    candidates: list[ModelRouteCandidate] = []
    for raw_candidate in raw_candidates:
        candidate = _strict_mapping(
            raw_candidate,
            {
                "allowed_sensitivities",
                "base_url",
                "profile",
                "provider_id",
                "secret_reference",
            },
        )
        raw_allowed = candidate["allowed_sensitivities"]
        if not isinstance(raw_allowed, list) or not raw_allowed:
            raise ValueError("allowed sensitivities are invalid")
        reference_value = candidate["secret_reference"]
        if reference_value is not None and not isinstance(reference_value, str):
            raise TypeError("secret reference must be text or null")
        base_url = candidate["base_url"]
        if base_url is not None and not isinstance(base_url, str):
            raise TypeError("base URL must be text or null")
        provider_id = candidate["provider_id"]
        if not isinstance(provider_id, str):
            raise TypeError("provider ID must be text")
        profile = _decode_model_profile(candidate["profile"])
        provider_name, separator, model_name = provider_id.partition(":")
        if not separator:
            raise ValueError("provider ID is incomplete")
        _, _, admitted_url, requires_credential = _admit_model_selection(
            provider_name,
            model_name,
            base_url,
        )
        reviewed_profile = reviewed_model_profile(provider_id)
        expected_profile = (
            reviewed_profile
            if reviewed_profile is not None
            else _model_profile(
                provider_name,
                model_name,
                context_window_tokens=profile.context_window_tokens,
                max_output_tokens=profile.max_output_tokens,
            )
        )
        if profile != expected_profile:
            raise AgentModelConfigurationError(
                "saved model configuration must be replaced"
            )
        reference = (
            None if reference_value is None else SecretReference.parse(reference_value)
        )
        if requires_credential and (
            reference is None or reference.scheme != "keychain"
        ):
            raise ValueError("persisted provider credential reference is incomplete")
        if requires_credential and (
            reference is None
            or not _credential_reference_is_owned(
                reference,
                agent_id=agent_id,
                provider=provider_name,
            )
        ):
            raise ValueError(
                "persisted provider credential reference belongs to another agent"
            )
        if not requires_credential and reference is not None:
            raise ValueError(
                "local or client-owned subscription configuration cannot contain "
                "a credential"
            )
        allowed = tuple(ModelSensitivity(cast(str, item)) for item in raw_allowed)
        if len(allowed) != len(set(allowed)):
            raise ValueError("allowed sensitivities cannot repeat")
        candidates.append(
            ModelRouteCandidate(
                provider_id=provider_id,
                profile=profile,
                base_url=admitted_url,
                secret_reference=reference,
                allowed_sensitivities=frozenset(allowed),
            )
        )
    raw_retry = _strict_mapping(
        route_value["retry_policy"],
        {"attempts", "backoff_seconds"},
    )
    route = ModelRoute(
        tuple(candidates),
        RetryPolicy(
            attempts=cast(int, raw_retry["attempts"]),
            backoff_seconds=cast(float, raw_retry["backoff_seconds"]),
        ),
    )
    raw_limits = _strict_mapping(
        document["limits"],
        {
            "max_estimated_cost_usd",
            "max_steps",
            "max_total_tokens",
            "max_wall_time_seconds",
        },
    )
    cost = raw_limits["max_estimated_cost_usd"]
    if cost is not None and not isinstance(cost, str):
        raise TypeError("estimated cost must be text or null")
    return AgentConfig(
        model_route=route,
        limits=LoopLimits(
            max_steps=cast(int, raw_limits["max_steps"]),
            max_total_tokens=cast(int, raw_limits["max_total_tokens"]),
            max_wall_time_seconds=cast(float, raw_limits["max_wall_time_seconds"]),
            max_estimated_cost_usd=None if cost is None else Decimal(cost),
        ),
    )


def _decode_model_profile(value: object) -> ModelProfile:
    fields = {
        "available",
        "context_window_tokens",
        "data_routing_classification",
        "healthy",
        "id",
        "input_cost_per_million_usd",
        "max_output_tokens",
        "output_cost_per_million_usd",
        "supports_documents",
        "supports_native_continuation",
        "supports_parallel_tools",
        "supports_prompt_caching",
        "supports_reasoning",
        "supports_streaming",
        "supports_structured_output",
        "supports_tools",
        "supports_vision",
    }
    profile = _strict_mapping(value, fields)
    if (
        profile["input_cost_per_million_usd"] is not None
        or profile["output_cost_per_million_usd"] is not None
    ):
        raise ValueError("model profile cannot contain pricing")
    return ModelProfile(
        id=cast(str, profile["id"]),
        context_window_tokens=cast(int, profile["context_window_tokens"]),
        max_output_tokens=cast(int, profile["max_output_tokens"]),
        supports_tools=cast(bool, profile["supports_tools"]),
        supports_parallel_tools=cast(bool, profile["supports_parallel_tools"]),
        supports_structured_output=cast(bool, profile["supports_structured_output"]),
        supports_streaming=cast(bool, profile["supports_streaming"]),
        supports_reasoning=cast(bool, profile["supports_reasoning"]),
        supports_vision=cast(bool, profile["supports_vision"]),
        supports_documents=cast(bool, profile["supports_documents"]),
        supports_prompt_caching=cast(bool, profile["supports_prompt_caching"]),
        supports_native_continuation=cast(
            bool, profile["supports_native_continuation"]
        ),
        data_routing_classification=cast(str, profile["data_routing_classification"]),
        available=cast(bool, profile["available"]),
        healthy=cast(bool, profile["healthy"]),
    )


def _strict_mapping(value: object, fields: set[str]) -> Mapping[str, object]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("configuration object fields are invalid")
    if any(not isinstance(key, str) for key in value):
        raise TypeError("configuration object keys must be text")
    return cast(Mapping[str, object], value)


def _validate_custom_loop(
    model: ModelProvider | None,
    profile: ModelProfile | None,
    context: ContextBuilder | None,
    tools: ToolRuntime | None,
) -> None:
    del profile
    if (context is None) != (tools is None):
        raise AgentNotConfiguredError(
            "custom context_builder and tools must be configured together"
        )
    if model is None and context is not None:
        raise AgentNotConfiguredError("custom loop components require a model")


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


def _admit_agent_home(
    name: str,
    root: str | Path | None,
    create: bool,
) -> tuple[Path, _WriterLock]:
    if not isinstance(name, str) or _AGENT_NAME.fullmatch(name) is None:
        raise AgentNameError(
            "agent name must contain 1-64 ASCII letters, digits, '_' or '-'"
        )
    state_root = _resolve_state_root(root)
    _reject_legacy_state_root(state_root)
    state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    agents_root = state_root / "agents"
    agents_root.mkdir(mode=0o700, exist_ok=True)
    home = agents_root / name
    if not create and not home.is_dir():
        raise AgentNotFoundError(f"agent does not exist: {name}")
    home.mkdir(mode=0o700, exist_ok=True)
    run = home / "run"
    run.mkdir(mode=0o700, exist_ok=True)
    for path, label in (
        (state_root, "agent state root"),
        (agents_root, "agents directory"),
        (home, "agent home"),
        (run, "agent run directory"),
    ):
        _require_unaliased_path(path, label)
        os.chmod(path, 0o700)
    return home, _WriterLock.acquire(run / "host.lock")


def _candidate_agent_homes(root: str | Path | None) -> tuple[Path, ...]:
    state_root = _resolve_state_root(root)
    if not state_root.exists():
        return ()
    if not state_root.is_dir():
        raise AgentHomeError("agent state root must be a directory")
    _reject_legacy_state_root(state_root)
    agents_root = state_root / "agents"
    if not agents_root.exists():
        return ()
    agents_root = _require_unaliased_path(agents_root, "agents directory")
    if not agents_root.is_dir():
        raise AgentHomeError("agents directory must be a directory")
    try:
        with os.scandir(agents_root) as iterator:
            entries = sorted(iterator, key=lambda entry: entry.name)
    except OSError as error:
        raise AgentHomeError("cannot inspect agents directory") from error
    candidates: list[Path] = []
    for entry in entries[:_MAX_AGENT_HOME_CANDIDATES]:
        if _AGENT_NAME.fullmatch(entry.name) is None:
            continue
        try:
            if not entry.is_dir(follow_symlinks=False):
                continue
            home = _require_unaliased_path(
                agents_root / entry.name,
                "agent home",
            )
        except (AgentHomeError, OSError):
            continue
        candidates.append(home)
    return tuple(candidates)


def _reject_legacy_state_root(state_root: Path) -> None:
    if (state_root / "agents").exists() or not state_root.exists():
        return
    if not any((state_root / marker).exists() for marker in _LEGACY_STATE_ROOT_MARKERS):
        return
    raise StateCompatibilityError(
        StateCompatibilityCode.LEGACY,
        state_root,
        (
            "This local data directory belongs to the unsupported pre-1.0 Daita "
            "framework. It was left unchanged. Use a separate current root "
            "and keep this directory intact for a deliberate legacy export/import."
        ),
        current_revision=SQLiteStateStore.current_revision,
        found_revision="pre-1.0-framework",
    )


def _resolve_state_root(root: str | Path | None) -> Path:
    state_root = Path.home() / ".daita" if root is None else Path(root)
    if ".." in state_root.parts:
        raise AgentHomeError("agent state root cannot contain parent aliases")
    state_root = Path(os.path.abspath(os.fspath(state_root)))
    return _require_unaliased_path(state_root, "agent state root")


def _require_unaliased_path(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    resolved = lexical.resolve(strict=False)
    if lexical != resolved or lexical.is_symlink():
        raise AgentHomeError(f"{label} cannot contain a symlink or path alias")
    return resolved


def _write_manifest(home: Path, identity: AgentIdentity) -> None:
    temporary = home / f".agent.toml.{uuid4().hex}.tmp"
    text = (
        f'agent_id = "{identity.id}"\n'
        f'display_name = "{identity.display_name}"\n'
        f'created_at = "{identity.created_at.isoformat()}"\n'
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, home / "agent.toml")
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _read_manifest(home: Path, expected_name: str) -> AgentIdentity:
    path = home / "agent.toml"
    state_path = home / "state.db"
    if path.is_symlink() or state_path.is_symlink():
        raise AgentIdentityMismatchError("agent files cannot be symlinks")
    if not path.is_file() or not state_path.is_file():
        raise AgentNotFoundError("agent home is incomplete")
    try:
        values = tomllib.loads(path.read_text(encoding="utf-8"))
        if set(values) != {"agent_id", "display_name", "created_at"}:
            raise ValueError("unexpected manifest fields")
        if values["display_name"] != expected_name:
            raise ValueError("manifest name differs from its home")
        return AgentIdentity(
            id=values["agent_id"],
            display_name=values["display_name"],
            created_at=datetime.fromisoformat(values["created_at"]),
        )
    except (OSError, ValueError, TypeError, tomllib.TOMLDecodeError) as error:
        raise AgentIdentityMismatchError("agent.toml is invalid") from error


def _cleanup_failed_create(home: Path) -> None:
    for name in (
        "agent.toml",
        "state.db",
        "state.db-wal",
        "state.db-shm",
        "state.db-journal",
    ):
        try:
            os.unlink(home / name)
        except FileNotFoundError:
            pass


def _delete_agent_home(home: Path) -> None:
    """Atomically retire one admitted home, then remove only that exact tree."""

    try:
        state = os.lstat(home)
    except OSError as error:
        raise AgentHomeError("agent home is unavailable for deletion") from error
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise AgentHomeError("agent home must be a non-symlink directory")
    if home.resolve(strict=True) != home:
        raise AgentHomeError("agent home cannot contain a symlink or path alias")
    tombstone = home.parent / f".{home.name}.deleting-{uuid4().hex}"
    try:
        os.replace(home, tombstone)
        directory = os.open(home.parent, os.O_RDONLY)
        try:
            try:
                os.fsync(directory)
            except OSError:
                pass
        finally:
            os.close(directory)
        shutil.rmtree(tombstone)
    except OSError as error:
        if tombstone.exists():
            raise AgentHomeError(
                "agent was retired but its temporary deletion tree remains"
            ) from error
        raise AgentHomeError("agent home could not be deleted") from error


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
    "SourceSelectionError",
]
