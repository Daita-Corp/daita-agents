"""Embedded composition for one isolated persistent agent home."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import errno
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import tomllib
from typing import Self, TypeVar
from uuid import uuid4

from .._json import FrozenJsonObject, canonical_json
from ..adapters.models import DiscoveryRequest, SourceRegistration
from ..adapters.protocols import ResourceAdapter, ResourceAdapterError, ResourceSource
from ..adapters.local_files import LocalDirectoryReadBackend
from ..adapters.sqlite import SQLiteSource
from ..adapters.sqlite_query import SQLiteQueryBackend
from ..adapters.postgresql_query import PostgreSQLQueryBackend
from ..adapters.sqlite_update import SQLiteUpdateBackend
from ..catalog.capabilities import catalog_declarations
from ..catalog.models import (
    CatalogResource,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSync,
    CatalogSyncStatus,
)
from ..catalog.service import CatalogService
from ..capabilities import CapabilityRegistry
from ..config import (
    AgentConfig,
    AgentRuntimeDefaults,
    AgentRuntimeDefaultsConflictError,
    resolve_agent_configuration,
)
from ..context import (
    MemoryContextProjector,
    SessionCompressionPolicy,
    SessionCompressionService,
    SkillContextProjector,
)
from ..domains.data import (
    CatalogDataView,
    DataContextBuilder,
    DataDomainController,
    PersistedAcceptedEvidenceDatasetReader,
    local_file_read_declarations,
    postgresql_query_declarations,
    sqlite_query_declarations,
    sqlite_update_declarations,
    tabular_comparison_declarations,
)
from ..events.models import CommittedEvent, EventCursor
from ..events.projection import EventAudience, project_committed_event
from ..errors import AgentError, ConfigError
from ..extensions import ConfiguredExtension, ExtensionBinding, ExtensionRegistry
from ..identity import AgentIdentity
from ..learning import (
    LearningProposal,
    LearningProposalState,
    LearningProvenance,
    LearningSourceOutcome,
)
from ..llm.models import ModelProfile
from ..llm.factory import create_model_route_provider, model_route_from_provider
from ..llm.protocols import (
    ModelProfileConflictError,
    ModelProvider,
    ModelRouteConflictError,
)
from ..llm.routing import ModelRoute
from ..loop.driver import AgentLoop, ContextBuilder, DomainController
from ..loop.models import LoopBudgets, LoopExit, LoopExitKind, LoopPhase
from ..memory.learning import (
    ExplicitCorrectionLearningError,
    ExplicitCorrectionLearningService,
    ExplicitCorrectionResult,
    is_explicit_learning_message,
)
from ..memory.models import (
    MemoryInspection,
    MemoryInspectionRequest,
    MemoryListRequest,
    MemoryListResult,
    MemoryRecallRequest,
    MemoryRecallResult,
    MemoryRestoreRequest,
    MemorySupersessionRequest,
)
from ..memory.service import MemoryService
from ..monitors.models import (
    Monitor,
    MonitorConfirmation,
    MonitorDefinition,
    MonitorInspection,
    MonitorProposal,
    MonitorStatus,
)
from ..monitors.service import MonitorService
from ..monitors.store import MonitorClaimResult
from ..operations.checkpoints import OperationSnapshot
from ..operations.governance import (
    ApprovalRequest,
    ApprovalStatus,
    DefaultPolicyEvaluator,
)
from ..operations.models import AgentTrigger, OperationStatus, TriggerKind
from ..operations.runtime import OperationRuntime
from ..sessions import Session, SessionAlreadyExistsError, SessionTranscript
from ..skills.models import SkillIndex, SkillInspection, SkillSource
from ..skills.learning import (
    SkillChangeAcceptanceResult,
    SkillChangeCandidate,
    SkillChangeLearningService,
    SkillChangeProposalResult,
)
from ..skills.service import SkillService
from ..storage.blobs import LocalBlobStore
from ..storage.sqlite import SQLiteOperationStore
from ..security import SecretProvider

_AGENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_T = TypeVar("_T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _source_sync_id(agent_id: str, idempotency_key: str) -> str:
    if (
        not isinstance(idempotency_key, str)
        or not idempotency_key
        or idempotency_key != idempotency_key.strip()
    ):
        raise ValueError("idempotency_key must be bounded non-empty text")
    try:
        encoded = idempotency_key.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError("idempotency_key must be valid UTF-8 text") from error
    if len(encoded) > 256:
        raise ValueError("idempotency_key exceeds 256 UTF-8 bytes")
    digest = sha256(
        canonical_json(
            {
                "agent_id": agent_id,
                "idempotency_key": idempotency_key,
                "owner": "source.attach",
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"catalog-sync-{digest}"


def _reject_protected_writable_source(
    registration: SourceRegistration,
    *,
    state_path: Path,
) -> None:
    """Keep controlled source writes outside the authoritative agent store."""

    if (
        registration.adapter_id != "sqlite"
        or registration.configuration.get("write_access") is not True
    ):
        return
    configured_path = registration.configuration.get("path")
    if not isinstance(configured_path, str):
        raise AgentHomeError("writable SQLite source path cannot be verified")
    try:
        source_stat = os.stat(configured_path, follow_symlinks=False)
        state_stat = os.stat(state_path, follow_symlinks=False)
    except OSError as error:
        raise AgentHomeError(
            "writable SQLite source path cannot be verified"
        ) from error
    if (source_stat.st_dev, source_stat.st_ino) == (
        state_stat.st_dev,
        state_stat.st_ino,
    ):
        raise AgentHomeError(
            "writable SQLite source cannot target protected agent state"
        )


def _reject_protected_sqlite_source_config(
    source: ResourceSource,
    *,
    state_path: Path,
) -> None:
    """Reject the built-in writable SQLite source before it opens a file."""

    if not isinstance(source, SQLiteSource) or not source.allow_writes:
        return
    try:
        source_stat = os.stat(source.path, follow_symlinks=False)
        state_stat = os.stat(state_path, follow_symlinks=False)
    except OSError:
        # The adapter remains the owner of ordinary path-admission failures.
        return
    if (source_stat.st_dev, source_stat.st_ino) == (
        state_stat.st_dev,
        state_stat.st_ino,
    ):
        raise AgentHomeError(
            "writable SQLite source cannot target protected agent state"
        )


def _validate_loop_configuration(
    model: ModelProvider | None,
    model_profile: ModelProfile | None,
    context_builder: ContextBuilder | None,
    domain: DomainController | None,
    capabilities: CapabilityRegistry | None,
    *,
    require_default_profile: bool,
) -> None:
    if model is None and (
        model_profile is not None or context_builder is not None or domain is not None
    ):
        raise AgentNotConfiguredError(
            "model_profile, context_builder, and domain require a configured model"
        )
    if (context_builder is None) != (domain is None):
        raise AgentNotConfiguredError(
            "context_builder and domain must be configured together"
        )
    if model is not None and context_builder is None and capabilities is not None:
        raise AgentNotConfiguredError(
            "custom capabilities require a custom context_builder and domain"
        )
    if model_profile is not None and not isinstance(model_profile, ModelProfile):
        raise TypeError("model_profile must be a ModelProfile or None")
    if (
        model is not None
        and model_profile is not None
        and model_profile.id != model.provider_id
    ):
        raise AgentNotConfiguredError(
            "model_profile id must match the configured model provider_id"
        )
    if model_profile is not None and (
        not model_profile.available or not model_profile.healthy
    ):
        raise AgentNotConfiguredError(
            "the configured model profile must be available and healthy"
        )
    if model_profile is not None and context_builder is not None:
        raise AgentNotConfiguredError(
            "a custom context_builder owns its own model-profile budgeting"
        )
    if (
        model_profile is not None
        and context_builder is None
        and not model_profile.supports_tools
    ):
        raise AgentNotConfiguredError(
            "the default data agent requires a tool-capable model profile"
        )
    if (
        require_default_profile
        and model is not None
        and model_profile is None
        and context_builder is None
    ):
        raise AgentNotConfiguredError(
            "the default data agent requires an explicit model profile"
        )


def _resolve_model_route_input(
    config: AgentConfig | None,
    model: ModelProvider | None,
    model_profile: ModelProfile | None,
    *,
    secret_provider: SecretProvider | None,
) -> tuple[ModelProvider | None, ModelProfile | None, ModelRoute | None]:
    explicit_route = None if config is None else config.model_route
    if explicit_route is not None:
        if model is not None:
            raise AgentNotConfiguredError(
                "an explicit model route reconstructs its provider; do not inject one"
            )
        if model_profile is not None and model_profile != explicit_route.model_profile:
            raise AgentNotConfiguredError(
                "configured model profile differs from the explicit model route"
            )
        return (
            create_model_route_provider(
                explicit_route,
                secret_provider=secret_provider,
            ),
            explicit_route.model_profile,
            explicit_route,
        )
    if model is None or model_profile is None:
        return model, model_profile, None
    if model.provider_id != model_profile.id:
        raise AgentNotConfiguredError(
            "model_profile id must match the configured model provider_id"
        )
    route = model_route_from_provider(model, model_profile)
    return model, model_profile, route


def _monitor_budgets_from_trigger(
    trigger: AgentTrigger,
    *,
    defaults: LoopBudgets,
) -> LoopBudgets:
    raw = trigger.payload.get("monitor_effective_budgets")
    expected_keys = {
        "max_actions",
        "max_estimated_cost_usd",
        "max_identical_failures",
        "max_observation_characters",
        "max_repairs",
        "max_total_tokens",
        "max_turns",
        "max_wall_time_seconds",
        "task_timeout_seconds",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_keys:
        raise ValueError("monitor trigger has no exact effective budget binding")
    cost_raw = raw["max_estimated_cost_usd"]
    try:
        cost = None if cost_raw is None else Decimal(str(cost_raw))
        budgets = LoopBudgets(
            max_turns=raw["max_turns"],
            max_actions=raw["max_actions"],
            max_repairs=raw["max_repairs"],
            max_identical_failures=raw["max_identical_failures"],
            max_observation_characters=raw["max_observation_characters"],
            max_total_tokens=raw["max_total_tokens"],
            max_wall_time_seconds=raw["max_wall_time_seconds"],
            task_timeout_seconds=raw["task_timeout_seconds"],
            max_estimated_cost_usd=cost,
        )
    except (InvalidOperation, TypeError, ValueError) as error:
        raise ValueError("monitor trigger effective budgets are invalid") from error
    if (
        budgets.max_turns > defaults.max_turns
        or budgets.max_actions > defaults.max_actions
        or budgets.max_wall_time_seconds > defaults.max_wall_time_seconds
        or budgets.max_repairs != defaults.max_repairs
        or budgets.max_identical_failures != defaults.max_identical_failures
        or budgets.max_observation_characters != defaults.max_observation_characters
        or budgets.max_total_tokens != defaults.max_total_tokens
        or budgets.task_timeout_seconds != defaults.task_timeout_seconds
        or budgets.max_estimated_cost_usd != defaults.max_estimated_cost_usd
    ):
        raise ValueError("monitor effective budgets may only restrict agent defaults")
    return budgets


async def _bind_or_load_runtime_defaults(
    store: SQLiteOperationStore,
    agent_id: str,
    *,
    policy: DefaultPolicyEvaluator | None,
    budgets: LoopBudgets | None,
) -> tuple[AgentRuntimeDefaults, DefaultPolicyEvaluator]:
    """Resolve future-operation defaults from the authoritative state store."""

    current = await store.load_runtime_defaults(agent_id)
    if current is None:
        active_policy = policy or DefaultPolicyEvaluator()
        proposed = AgentRuntimeDefaults(
            budgets=budgets or LoopBudgets(),
            policy_profile=active_policy.profile,
        )
        try:
            current = await store.bind_runtime_defaults(agent_id, proposed)
        except AgentRuntimeDefaultsConflictError as error:
            raise ConfigError(
                "Agent runtime defaults changed while they were being bound.",
                section="runtime_defaults",
                error_code="config_conflict",
            ) from error
        return current, active_policy
    if budgets is not None and budgets != current.budgets:
        raise ConfigError(
            "Configured budgets differ from the stored agent defaults.",
            section="budgets",
            error_code="config_conflict",
        )
    if policy is not None and policy.profile != current.policy_profile:
        raise ConfigError(
            "Configured policy differs from the stored agent defaults.",
            section="policy",
            error_code="config_conflict",
        )
    return current, policy or DefaultPolicyEvaluator(current.policy_profile)


async def _await_sync_completion(callback: Callable[[], _T]) -> tuple[_T, bool]:
    """Finish one filesystem transaction before propagating cancellation."""

    worker = asyncio.create_task(asyncio.to_thread(callback))
    cancellation_requested = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancellation_requested = True
            continue
        except BaseException:
            break
    try:
        result = worker.result()
    except BaseException as error:
        if cancellation_requested and not isinstance(error, asyncio.CancelledError):
            raise asyncio.CancelledError from error
        raise
    return result, cancellation_requested


class AgentHomeError(AgentError):
    """Base failure for isolated local agent-home admission."""


class AgentNameError(AgentHomeError, ValueError):
    """Raised when a name could escape or alias its agent directory."""


class AgentAlreadyExistsError(AgentHomeError):
    """Raised when create targets an initialized agent home."""


class AgentNotFoundError(AgentHomeError):
    """Raised when open targets no initialized agent home."""


class AgentIdentityMismatchError(AgentHomeError):
    """Raised when bootstrap and authoritative database identity differ."""


class HostActiveError(AgentHomeError):
    """Raised rather than opening a competing embedded writer."""

    code = "host_active"


class AgentNotConfiguredError(AgentHomeError):
    """Raised when run/resume has no composed loop dependencies."""


class SessionOperationActiveError(AgentHomeError):
    """Raised before creating a second operation in one active session."""

    code = "session_operation_active"

    def __init__(self, session_id: str, operation_id: str, status: str) -> None:
        self.session_id = session_id
        self.operation_id = operation_id
        self.status = status
        super().__init__(
            f"session_operation_active: session {session_id} already has "
            f"nonterminal operation {operation_id} ({status})"
        )


class _WriterLock:
    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self._descriptor = descriptor

    @classmethod
    def acquire(cls, path: Path) -> _WriterLock:
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(path, flags, 0o600)
        except OSError as error:
            raise AgentHomeError(f"cannot open agent writer lock: {path}") from error
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise AgentHomeError("agent writer lock must be a regular file")
            try:
                import fcntl

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
        try:
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


class EmbeddedAgent:
    """Compose existing runtime owners under one local writer admission."""

    def __init__(
        self,
        *,
        identity: AgentIdentity,
        home: Path,
        writer_lock: _WriterLock,
        store: SQLiteOperationStore,
        runtime: OperationRuntime,
        loop: AgentLoop | None,
        capabilities: CapabilityRegistry,
        model_profile: ModelProfile | None,
        model_route: ModelRoute | None,
        runtime_defaults: AgentRuntimeDefaults,
        catalog_service: CatalogService,
        memory_service: MemoryService,
        monitor_service: MonitorService,
        skill_service: SkillService,
        learning_service: ExplicitCorrectionLearningService,
        skill_change_service: SkillChangeLearningService,
        extension_registry: ExtensionRegistry,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        self.identity = identity
        self.home = home
        self._writer_lock = writer_lock
        self._store = store
        self._runtime = runtime
        self._loop = loop
        self._capabilities = capabilities
        self.model_profile = model_profile
        self.model_route = model_route
        self.runtime_defaults = runtime_defaults
        self._catalog_service = catalog_service
        self._memory_service = memory_service
        self._monitor_service = monitor_service
        self._skill_service = skill_service
        self._learning_service = learning_service
        self._skill_change_service = skill_change_service
        self._extension_registry = extension_registry
        self._clock = clock
        self._id_factory = id_factory
        self._mutation_lock = asyncio.Lock()
        self._closed = False

    @property
    def extension_bindings(self) -> tuple[ExtensionBinding, ...]:
        return self._extension_registry.bindings

    async def configure_model_route(
        self,
        route: ModelRoute,
        *,
        expected_revision: int,
    ) -> ModelRoute:
        """Bind a future-operation route; this composition must reopen to run."""

        if not isinstance(route, ModelRoute):
            raise TypeError("route must be a ModelRoute")
        async with self._mutation_lock:
            self._require_open()
            stored = await self._store.set_model_route(
                self.identity.id,
                route,
                expected_revision=expected_revision,
            )
            self.model_route = stored
            self.model_profile = stored.model_profile
            # Runtime and loop bind route identity at composition. Deliberately
            # disable this in-memory composition until a normal reopen rebuilds
            # them from the durable route.
            self._loop = None
            return stored

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
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        extensions: tuple[ConfiguredExtension, ...] = (),
    ) -> Self:
        extension_registry = ExtensionRegistry.load(extensions)
        if extension_registry.extension_ids and any(
            value is not None for value in (context_builder, domain, capabilities)
        ):
            raise AgentNotConfiguredError(
                "configured extensions require the default data composition"
            )
        model_profile, policy, budgets = resolve_agent_configuration(
            config,
            model=model,
            model_profile=model_profile,
            policy=policy,
            budgets=budgets,
        )
        model, model_profile, requested_route = _resolve_model_route_input(
            config,
            model,
            model_profile,
            secret_provider=secret_provider,
        )
        _validate_loop_configuration(
            model,
            model_profile,
            context_builder,
            domain,
            capabilities,
            require_default_profile=True,
        )
        resolved_clock = _utc_now if clock is None else clock
        resolved_id_factory = _new_id if id_factory is None else id_factory
        admission, admission_cancelled = await _await_sync_completion(
            lambda: _admit_agent_home(name, root, True)
        )
        home, writer_lock = admission
        if admission_cancelled:
            writer_lock.release()
            raise asyncio.CancelledError
        store: SQLiteOperationStore | None = None
        bootstrap_started = False
        bootstrap_published = False
        try:
            manifest_path = home / "agent.toml"
            state_path = home / "state.db"
            if (
                manifest_path.exists()
                or manifest_path.is_symlink()
                or state_path.exists()
                or state_path.is_symlink()
            ):
                raise AgentAlreadyExistsError(f"agent already exists: {name}")
            bootstrap_started = True
            created_at = resolved_clock()
            identity = AgentIdentity(
                id=resolved_id_factory("agent"),
                display_name=name,
                created_at=created_at,
            )
            store = await SQLiteOperationStore.open(
                state_path,
                clock=resolved_clock,
            )
            await store.initialize_identity(identity)
            await store.bind_extension_bindings(
                identity.id,
                extension_registry.bindings,
            )
            resolved_profile = (
                model_profile if model is not None and context_builder is None else None
            )
            resolved_route = (
                requested_route
                if resolved_profile is not None and context_builder is None
                else None
            )
            if resolved_route is not None:
                try:
                    resolved_route = await store.set_model_route(
                        identity.id,
                        resolved_route,
                        expected_revision=0,
                    )
                except ModelRouteConflictError as error:
                    raise AgentNotConfiguredError(
                        "agent model route changed during creation"
                    ) from error
            elif resolved_profile is not None:
                try:
                    resolved_profile = await store.bind_model_profile(
                        identity.id,
                        resolved_profile,
                    )
                except ModelProfileConflictError as error:
                    raise AgentNotConfiguredError(
                        "agent model profile changed during creation"
                    ) from error
            runtime_defaults, active_policy = await _bind_or_load_runtime_defaults(
                store,
                identity.id,
                policy=policy,
                budgets=budgets,
            )
            embedded = cls._compose(
                identity=identity,
                home=home,
                writer_lock=writer_lock,
                store=store,
                model=model,
                model_profile=resolved_profile,
                model_route=resolved_route,
                context_builder=context_builder,
                domain=domain,
                capabilities=capabilities,
                policy=active_policy,
                budgets=runtime_defaults.budgets,
                runtime_defaults=runtime_defaults,
                clock=resolved_clock,
                id_factory=resolved_id_factory,
                secret_provider=secret_provider,
                extension_registry=extension_registry,
            )
            _, manifest_cancelled = await _await_sync_completion(
                lambda: _write_manifest(home, identity)
            )
            bootstrap_published = True
            if manifest_cancelled:
                raise asyncio.CancelledError
            return embedded
        except BaseException:
            try:
                if store is not None:
                    await store.close()
            finally:
                try:
                    if bootstrap_started and not bootstrap_published:
                        await _await_sync_completion(
                            lambda: _cleanup_failed_create(home)
                        )
                finally:
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
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets | None = None,
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
        secret_provider: SecretProvider | None = None,
        extensions: tuple[ConfiguredExtension, ...] = (),
    ) -> Self:
        extension_registry = ExtensionRegistry.load(extensions)
        if extension_registry.extension_ids and any(
            value is not None for value in (context_builder, domain, capabilities)
        ):
            raise AgentNotConfiguredError(
                "configured extensions require the default data composition"
            )
        model_profile, policy, budgets = resolve_agent_configuration(
            config,
            model=model,
            model_profile=model_profile,
            policy=policy,
            budgets=budgets,
        )
        model, model_profile, requested_route = _resolve_model_route_input(
            config,
            model,
            model_profile,
            secret_provider=secret_provider,
        )
        _validate_loop_configuration(
            model,
            model_profile,
            context_builder,
            domain,
            capabilities,
            require_default_profile=False,
        )
        resolved_clock = _utc_now if clock is None else clock
        resolved_id_factory = _new_id if id_factory is None else id_factory
        admission, admission_cancelled = await _await_sync_completion(
            lambda: _admit_agent_home(name, root, False)
        )
        home, writer_lock = admission
        if admission_cancelled:
            writer_lock.release()
            raise asyncio.CancelledError
        store: SQLiteOperationStore | None = None
        try:
            manifest_result, manifest_cancelled = await _await_sync_completion(
                lambda: _read_manifest(home, name)
            )
            manifest = manifest_result
            if manifest_cancelled:
                raise asyncio.CancelledError
            store = await SQLiteOperationStore.open(
                home / "state.db",
                clock=resolved_clock,
            )
            identity = await store.load_identity()
            if identity is None or identity != manifest:
                raise AgentIdentityMismatchError(
                    "agent.toml does not match authoritative state.db identity"
                )
            extension_registry.validate_binding(
                await store.load_extension_bindings(identity.id)
            )
            stored_route = await store.load_model_route(identity.id)
            stored_profile = await store.load_model_profile(identity.id)
            resolved_profile: ModelProfile | None
            resolved_route: ModelRoute | None = None
            if stored_route is not None:
                if context_builder is not None or domain is not None:
                    raise AgentNotConfiguredError(
                        "a persisted model route requires the default data composition"
                    )
                if requested_route is not None and requested_route != stored_route:
                    try:
                        stored_route = await store.set_model_route(
                            identity.id,
                            requested_route,
                            expected_revision=stored_route.revision,
                        )
                    except ModelRouteConflictError as error:
                        raise AgentNotConfiguredError(
                            "configured model route conflicts with durable state"
                        ) from error
                resolved_route = stored_route
                resolved_profile = stored_route.model_profile
                if model_profile is not None and model_profile != resolved_profile:
                    raise AgentNotConfiguredError(
                        "configured model profile differs from the stored route"
                    )
                if requested_route is None:
                    if model is None:
                        model = create_model_route_provider(
                            stored_route,
                            secret_provider=secret_provider,
                        )
                    else:
                        proposed = model_route_from_provider(
                            model,
                            resolved_profile,
                            revision=stored_route.revision,
                        )
                        if proposed != stored_route:
                            raise AgentNotConfiguredError(
                                "configured model provider differs from the stored route"
                            )
                else:
                    model = create_model_route_provider(
                        stored_route,
                        secret_provider=secret_provider,
                    )
            elif requested_route is not None:
                try:
                    resolved_route = await store.set_model_route(
                        identity.id,
                        requested_route,
                        expected_revision=0,
                    )
                except ModelRouteConflictError as error:
                    raise AgentNotConfiguredError(
                        "configured model route conflicts with durable state"
                    ) from error
                resolved_profile = resolved_route.model_profile
                model = create_model_route_provider(
                    resolved_route,
                    secret_provider=secret_provider,
                )
            elif model is None:
                resolved_profile = stored_profile
            elif context_builder is not None:
                if (
                    stored_profile is not None
                    and stored_profile.id != model.provider_id
                ):
                    raise AgentNotConfiguredError(
                        "configured model provider differs from the stored profile"
                    )
                resolved_profile = None
            elif stored_profile is None:
                if model_profile is None:
                    raise AgentNotConfiguredError(
                        "the default data agent requires an explicit model profile "
                        "on its first configured open"
                    )
                try:
                    resolved_profile = await store.bind_model_profile(
                        identity.id,
                        model_profile,
                    )
                except ModelProfileConflictError as error:
                    raise AgentNotConfiguredError(
                        "agent model profile changed during open"
                    ) from error
            else:
                if stored_profile.id != model.provider_id:
                    raise AgentNotConfiguredError(
                        "configured model provider differs from the stored profile"
                    )
                if model_profile is not None and model_profile != stored_profile:
                    raise AgentNotConfiguredError(
                        "configured model profile differs from the stored profile"
                    )
                resolved_profile = stored_profile
            if model is not None and context_builder is None:
                _validate_loop_configuration(
                    model,
                    resolved_profile,
                    context_builder,
                    domain,
                    capabilities,
                    require_default_profile=True,
                )
            runtime_defaults, active_policy = await _bind_or_load_runtime_defaults(
                store,
                identity.id,
                policy=policy,
                budgets=budgets,
            )
            return cls._compose(
                identity=identity,
                home=home,
                writer_lock=writer_lock,
                store=store,
                model=model,
                model_profile=resolved_profile,
                model_route=resolved_route,
                context_builder=context_builder,
                domain=domain,
                capabilities=capabilities,
                policy=active_policy,
                budgets=runtime_defaults.budgets,
                runtime_defaults=runtime_defaults,
                clock=resolved_clock,
                id_factory=resolved_id_factory,
                secret_provider=secret_provider,
                extension_registry=extension_registry,
            )
        except BaseException:
            try:
                if store is not None:
                    await store.close()
            finally:
                writer_lock.release()
            raise

    @classmethod
    def _compose(
        cls,
        *,
        identity: AgentIdentity,
        home: Path,
        writer_lock: _WriterLock,
        store: SQLiteOperationStore,
        model: ModelProvider | None,
        model_profile: ModelProfile | None,
        model_route: ModelRoute | None,
        context_builder: ContextBuilder | None,
        domain: DomainController | None,
        capabilities: CapabilityRegistry | None,
        policy: DefaultPolicyEvaluator | None,
        budgets: LoopBudgets,
        runtime_defaults: AgentRuntimeDefaults,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
        secret_provider: SecretProvider | None,
        extension_registry: ExtensionRegistry,
    ) -> Self:
        resolved_context = context_builder
        resolved_domain = domain
        resolved_capabilities = capabilities
        blob_store = LocalBlobStore(home / "blobs")
        catalog_service = CatalogService(store)
        memory_service = MemoryService(store, clock=clock)
        monitor_service = MonitorService(
            agent_id=identity.id,
            store=store,
            default_budgets=runtime_defaults.budgets,
            default_policy=runtime_defaults.policy_profile,
            clock=clock,
            id_factory=id_factory,
        )
        resolved_profile = model_profile
        data_view: CatalogDataView | None = None
        if context_builder is None and domain is None and capabilities is None:
            data_view = CatalogDataView(store, catalog_service)
            catalog = catalog_declarations(identity.id, catalog_service)
            sqlite_query = sqlite_query_declarations(
                identity.id,
                SQLiteQueryBackend(store, data_view),
            )
            postgresql_query = postgresql_query_declarations(
                identity.id,
                PostgreSQLQueryBackend(
                    store,
                    data_view,
                    secret_provider,
                ),
            )
            update = sqlite_update_declarations(
                identity.id,
                SQLiteUpdateBackend(
                    store,
                    data_view,
                    store,
                    protected_paths=(home / "state.db",),
                ),
            )
            file_read = local_file_read_declarations(
                identity.id,
                LocalDirectoryReadBackend(store, store),
            )
            comparison = tabular_comparison_declarations(
                PersistedAcceptedEvidenceDatasetReader(
                    store,
                    store,
                    store,
                    blob_store,
                )
            )
            built_in_capabilities = CapabilityRegistry(
                capabilities=(
                    *catalog.capabilities,
                    *sqlite_query.capabilities,
                    *postgresql_query.capabilities,
                    *update.capabilities,
                    *file_read.capabilities,
                    *comparison.capabilities,
                ),
                executors=(
                    *catalog.executors,
                    *sqlite_query.executors,
                    *postgresql_query.executors,
                    *update.executors,
                    *file_read.executors,
                    *comparison.executors,
                ),
                tool_views=(
                    *catalog.tool_views,
                    *sqlite_query.tool_views,
                    *postgresql_query.tool_views,
                    *update.tool_views,
                    *file_read.tool_views,
                    *comparison.tool_views,
                ),
            )
            resolved_capabilities = extension_registry.compose_with(
                built_in_capabilities
            )
        active_capabilities = resolved_capabilities or CapabilityRegistry()
        skills_root = _ensure_agent_directory(home, "skills")
        skill_service = SkillService(
            agent_id=identity.id,
            root=skills_root,
            source=SkillSource.USER,
            store=store,
            capability_ids=active_capabilities.capability_ids,
            clock=clock,
            id_factory=id_factory,
        )
        learning_service = ExplicitCorrectionLearningService(
            catalog=store,
            store=store,
            clock=clock,
        )
        skill_change_service = SkillChangeLearningService(
            agent_id=identity.id,
            store=store,
            skills=skill_service,
            clock=clock,
            id_factory=id_factory,
        )
        if model is not None and data_view is not None:
            assert resolved_profile is not None
            if not resolved_profile.supports_tools:
                raise AgentNotConfiguredError(
                    "the default data agent requires a tool-capable model profile"
                )
            session_context = SessionCompressionService(
                transcripts=store,
                checkpoints=store,
                operations=store,
                committer=store,
                policy=SessionCompressionPolicy(
                    compression_threshold_tokens=max(
                        1,
                        resolved_profile.maximum_input_tokens * 3 // 4,
                    )
                ),
                clock=clock,
                id_factory=id_factory,
            )
            resolved_context = DataContextBuilder(
                data_view,
                profile=resolved_profile,
                session_projector=session_context,
                memory_projector=MemoryContextProjector(memory_service),
                skill_projector=SkillContextProjector(skill_service),
            )
            resolved_domain = DataDomainController(
                active_capabilities,
                data_view,
                clock=clock,
            )
        runtime = OperationRuntime(
            clock=clock,
            id_factory=id_factory,
            capabilities=active_capabilities,
            store=store,
            blob_store=blob_store,
            policy=policy,
            model_route_revision=(
                None if model_route is None else model_route.revision
            ),
            model_route_fingerprint=(
                None if model_route is None else model_route.fingerprint
            ),
        )
        loop = (
            None
            if model is None or resolved_context is None or resolved_domain is None
            else AgentLoop(
                runtime=runtime,
                model=model,
                context_builder=resolved_context,
                domain=resolved_domain,
                budgets=budgets,
            )
        )
        return cls(
            identity=identity,
            home=home,
            writer_lock=writer_lock,
            store=store,
            runtime=runtime,
            loop=loop,
            capabilities=active_capabilities,
            model_profile=resolved_profile,
            model_route=model_route,
            runtime_defaults=runtime_defaults,
            catalog_service=catalog_service,
            memory_service=memory_service,
            monitor_service=monitor_service,
            skill_service=skill_service,
            learning_service=learning_service,
            skill_change_service=skill_change_service,
            extension_registry=extension_registry,
            clock=clock,
            id_factory=id_factory,
        )

    async def run(self, message: str, *, session_id: str | None = None) -> LoopExit:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        trigger_id = self._id_factory("trigger")
        return await self.run_trigger(
            AgentTrigger(
                id=trigger_id,
                agent_id=self.identity.id,
                kind=TriggerKind.USER,
                source_id=f"user:{session_id or trigger_id}",
                session_id=session_id,
                payload={"message": message},
                created_at=self._clock(),
            )
        )

    async def stream(
        self,
        message: str,
        *,
        session_id: str | None = None,
    ) -> AsyncGenerator[FrozenJsonObject, None]:
        """Run one user trigger and yield its public committed-event views."""

        self._require_loop()
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        trigger_id = self._id_factory("trigger")
        trigger = AgentTrigger(
            id=trigger_id,
            agent_id=self.identity.id,
            kind=TriggerKind.USER,
            source_id=f"user:{session_id or trigger_id}",
            session_id=session_id,
            payload={"message": message},
            created_at=self._clock(),
        )
        cursor = await self._store.latest_cursor(self.identity.id)
        subscription = self._store.subscribe(self.identity.id, cursor)
        run_task = asyncio.create_task(
            self.run_trigger(trigger),
            name=f"daita-agent-stream:{trigger_id}",
        )
        next_event_task: asyncio.Task[CommittedEvent] | None = None
        operation_id: str | None = None
        target_event_id: str | None = None
        last_yielded_event_id: str | None = None
        try:
            while True:
                if next_event_task is None:
                    next_event_task = asyncio.create_task(anext(subscription))
                waiters: set[asyncio.Task[object]] = {next_event_task}
                if target_event_id is None:
                    waiters.add(run_task)
                completed, _ = await asyncio.wait(
                    waiters,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if run_task in completed and target_event_id is None:
                    result = run_task.result()
                    operation_id = result.operation_id
                    snapshot = (await self._store.load(operation_id)).snapshot
                    if not snapshot.events:
                        raise AgentHomeError(
                            "streamed operation has no committed runtime events"
                        )
                    target_event_id = snapshot.events[-1].id
                    if last_yielded_event_id == target_event_id:
                        return
                if next_event_task not in completed:
                    continue
                committed = next_event_task.result()
                next_event_task = None
                if operation_id is None:
                    claimed = await self._store.load_by_trigger(trigger.id)
                    if claimed is not None:
                        operation_id = claimed.snapshot.operation.id
                if committed.event.operation_id != operation_id:
                    continue
                last_yielded_event_id = committed.event.id
                yield project_committed_event(
                    committed,
                    audience=EventAudience.PUBLIC,
                )
                if last_yielded_event_id == target_event_id:
                    return
        finally:
            if next_event_task is not None and not next_event_task.done():
                next_event_task.cancel()
                try:
                    await next_event_task
                except (asyncio.CancelledError, StopAsyncIteration):
                    pass
            await subscription.aclose()
            if not run_task.done():
                run_task.cancel()
                try:
                    await run_task
                except asyncio.CancelledError:
                    pass
            else:
                try:
                    run_task.exception()
                except asyncio.CancelledError:
                    pass

    async def run_trigger(self, trigger: AgentTrigger) -> LoopExit:
        """Run one exact durable trigger through the single canonical loop."""

        loop = self._require_loop()
        if not isinstance(trigger, AgentTrigger):
            raise TypeError("trigger must be an AgentTrigger")
        if trigger.agent_id != self.identity.id:
            raise ValueError("trigger belongs to another agent")
        async with self._mutation_lock:
            self._require_open()
            if trigger.kind is TriggerKind.USER:
                message = trigger.payload.get("message")
                if not isinstance(message, str) or not message.strip():
                    raise ValueError(
                        "user trigger payload must contain a non-empty message"
                    )
                if trigger.session_id is not None:
                    existing = await self._store.load_by_trigger(trigger.id)
                    if existing is None:
                        await self._ensure_session(trigger.session_id, message)
                    elif existing.snapshot.trigger != trigger:
                        raise ValueError(
                            "trigger identity already owns different durable input"
                        )
            effective_budgets = (
                self.runtime_defaults.budgets
                if trigger.kind is not TriggerKind.MONITOR
                else _monitor_budgets_from_trigger(
                    trigger,
                    defaults=self.runtime_defaults.budgets,
                )
            )
            result = await loop.run(trigger, budgets=effective_budgets)
            return await self._with_post_operation_learning(result)

    async def attach(
        self,
        source: ResourceSource,
        *,
        idempotency_key: str | None = None,
    ) -> SourceRegistration:
        """Run one bounded source discovery and commit its complete catalog view."""

        if not isinstance(source, ResourceSource):
            raise TypeError("source must provide async open(...)")
        stable_sync_id = (
            None
            if idempotency_key is None
            else _source_sync_id(self.identity.id, idempotency_key)
        )
        async with self._mutation_lock:
            self._require_open()
            _reject_protected_sqlite_source_config(
                source,
                state_path=self.home / "state.db",
            )
            opened_at = self._clock()
            adapter = await source.open(
                agent_id=self.identity.id,
                attached_at=opened_at,
                clock=self._clock,
            )
            if not isinstance(adapter, ResourceAdapter):
                raise TypeError("source open() must return a ResourceAdapter")
            registration = adapter.registration
            try:
                _reject_protected_writable_source(
                    registration,
                    state_path=self.home / "state.db",
                )
                self._capabilities.validate_declarations(adapter.declarations())
            except AgentHomeError:
                await adapter.close()
                raise
            except (TypeError, ValueError) as error:
                await adapter.close()
                raise AgentHomeError(
                    "source declarations do not match the configured runtime"
                ) from error
            sync_id = (
                self._id_factory("catalog-sync")
                if stable_sync_id is None
                else stable_sync_id
            )
            try:
                existing_sync = await self._store.load_sync(
                    self.identity.id,
                    sync_id,
                )
                started_at = (
                    opened_at if existing_sync is None else existing_sync.started_at
                )
                running = CatalogSync(
                    id=sync_id,
                    agent_id=self.identity.id,
                    source_id=registration.id,
                    adapter_id=registration.adapter_id,
                    status=CatalogSyncStatus.RUNNING,
                    started_at=started_at,
                )
            except BaseException:
                await adapter.close()
                raise
            try:
                if existing_sync is not None:
                    if (
                        existing_sync.source_id != registration.id
                        or existing_sync.adapter_id != registration.adapter_id
                    ):
                        raise AgentHomeError(
                            "source idempotency key is bound to another source"
                        )
                    if existing_sync.status is CatalogSyncStatus.SUCCEEDED:
                        replayed = await self._store.load_source(
                            self.identity.id,
                            registration.id,
                        )
                        if (
                            replayed is None
                            or not replayed.active
                            or replayed.adapter_id != registration.adapter_id
                            or replayed.native_identity != registration.native_identity
                            or replayed.configuration != registration.configuration
                        ):
                            raise AgentHomeError(
                                "successful source attachment has inconsistent state"
                            )
                        return replayed
                    if existing_sync.status is not CatalogSyncStatus.RUNNING:
                        raise AgentHomeError(
                            "source attachment idempotency key names a failed sync"
                        )
                await self._store.record_sync(running)
                result = await adapter.discover(
                    DiscoveryRequest(
                        agent_id=self.identity.id,
                        source_id=registration.id,
                        sync_id=sync_id,
                        requested_at=started_at,
                    )
                )
                existing = await self._store.load_source(
                    self.identity.id,
                    registration.id,
                )
                if existing is None:
                    committed_registration = await self._store.register_source(
                        registration
                    )
                elif (
                    existing.active
                    and existing.adapter_id == registration.adapter_id
                    and existing.native_identity == registration.native_identity
                    and existing.configuration == registration.configuration
                ):
                    committed_registration = existing
                else:
                    raise AgentHomeError(
                        f"source registration conflicts with existing source: "
                        f"{registration.id}"
                    )
                await self._store.commit_snapshot(result.snapshot)
                return committed_registration
            except BaseException as error:
                failed_at = self._clock()
                error_code = (
                    error.code
                    if isinstance(error, ResourceAdapterError)
                    else "source_attach_failed"
                )
                failed = CatalogSync(
                    id=sync_id,
                    agent_id=self.identity.id,
                    source_id=registration.id,
                    adapter_id=registration.adapter_id,
                    status=CatalogSyncStatus.FAILED,
                    started_at=started_at,
                    completed_at=max(failed_at, started_at),
                    error_code=error_code,
                )
                try:
                    await self._store.record_sync(failed)
                except BaseException:
                    pass
                raise
            finally:
                await adapter.close()

    async def detach(self, source_id: str) -> SourceRegistration:
        """Persist the source store's one-way detach transition."""

        async with self._mutation_lock:
            self._require_open()
            return await self._store.detach_source(
                self.identity.id,
                source_id,
                self._clock(),
            )

    async def list_sources(self) -> tuple[SourceRegistration, ...]:
        self._require_open()
        return await self._store.list_sources(self.identity.id)

    async def list_catalog_resources(
        self,
        *,
        source_id: str | None = None,
    ) -> tuple[CatalogResource, ...]:
        self._require_open()
        return await self._store.list_resources(self.identity.id, source_id)

    async def search_catalog(
        self,
        request: CatalogSearchRequest,
    ) -> CatalogSearchResult:
        self._require_open()
        if not isinstance(request, CatalogSearchRequest):
            raise TypeError("request must be a CatalogSearchRequest")
        if request.agent_id != self.identity.id:
            raise ValueError("catalog search belongs to another agent")
        return await self._catalog_service.search(request)

    async def inspect_catalog_resource(self, resource_id: str) -> FrozenJsonObject:
        self._require_open()
        return await self._catalog_service.inspect_resource(
            self.identity.id,
            resource_id,
        )

    async def inspect(self, operation_id: str) -> OperationSnapshot:
        self._require_open()
        return (await self._store.load(operation_id)).snapshot

    async def list_operations(
        self,
        *,
        statuses: tuple[OperationStatus, ...] | None = None,
        limit: int = 100,
    ) -> tuple[OperationSnapshot, ...]:
        self._require_open()
        operations = await self._store.list_operations(
            self.identity.id,
            statuses=statuses,
            limit=limit,
        )
        return tuple(operation.snapshot for operation in operations)

    async def inspect_approval(self, approval_id: str) -> ApprovalRequest:
        self._require_open()
        operation = await self._store.load_by_approval(approval_id)
        if (
            operation is None
            or operation.snapshot.operation.agent_id != self.identity.id
        ):
            raise KeyError(f"unknown approval: {approval_id}")
        return next(
            approval
            for approval in operation.snapshot.approvals
            if approval.id == approval_id
        )

    async def list_approvals(
        self,
        *,
        statuses: tuple[ApprovalStatus, ...] | None = None,
        limit: int = 100,
    ) -> tuple[ApprovalRequest, ...]:
        self._require_open()
        if statuses is not None:
            statuses = tuple(statuses)
            if not statuses or any(
                not isinstance(status, ApprovalStatus) for status in statuses
            ):
                raise ValueError("approval statuses must contain ApprovalStatus values")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= 1_000
        ):
            raise ValueError("approval list limit must be from one through 1000")
        operations = await self._store.list_operations(self.identity.id, limit=1_000)
        approvals = tuple(
            approval
            for operation in operations
            for approval in operation.snapshot.approvals
            if statuses is None or approval.status in statuses
        )
        return tuple(
            sorted(
                approvals,
                key=lambda approval: (approval.requested_at, approval.id),
                reverse=True,
            )[:limit]
        )

    async def resume(self, operation_id: str) -> LoopExit:
        loop = self._require_loop()
        async with self._mutation_lock:
            self._require_open()
            result = await loop.resume(operation_id)
            return await self._with_post_operation_learning(result)

    async def recover_startup(self) -> tuple[LoopExit, ...]:
        """Resume one ordered snapshot of this agent's recoverable operations."""

        loop = self._require_loop()
        async with self._mutation_lock:
            self._require_open()
            recovered = await loop.recover_startup(self.identity.id)
            exits: list[LoopExit] = []
            for result in recovered:
                exits.append(await self._with_post_operation_learning(result))
            return tuple(exits)

    async def decide_approval(
        self,
        approval_id: str,
        *,
        status: ApprovalStatus,
        decided_by: str,
        reason: str,
    ) -> ApprovalRequest:
        """Persist an approval decision without executing or resuming work."""

        self._require_open()
        return await self._runtime.decide_approval(
            approval_id,
            status=status,
            decided_by=decided_by,
            reason=reason,
        )

    async def interrupt(
        self,
        operation_id: str,
        reason: str = "user_cancelled",
    ) -> LoopExit:
        """Persist runtime-owned interruption without waiting for loop admission."""

        self._require_open()
        return await self._runtime.interrupt(operation_id, reason)

    async def inspect_nonterminal(self) -> tuple[OperationSnapshot, ...]:
        self._require_open()
        return await self._runtime.inspect_nonterminal(self.identity.id)

    async def read_events(
        self,
        cursor: EventCursor | None = None,
        *,
        limit: int = 100,
    ) -> tuple[CommittedEvent, ...]:
        self._require_open()
        return await self._store.read_after(
            self.identity.id,
            cursor,
            limit=limit,
        )

    def subscribe_events(
        self,
        cursor: EventCursor | None = None,
    ) -> AsyncGenerator[CommittedEvent, None]:
        self._require_open()
        return self._store.subscribe(self.identity.id, cursor)

    async def propose_monitor(
        self,
        monitor_id: str,
        definition: MonitorDefinition,
        *,
        idempotency_key: str,
        source_operation_id: str | None = None,
    ) -> MonitorProposal:
        async with self._mutation_lock:
            self._require_open()
        return await self._monitor_service.propose(
            monitor_id,
            definition,
            idempotency_key=idempotency_key,
            source_operation_id=source_operation_id,
        )

    async def propose_monitor_natural(
        self,
        monitor_id: str,
        request: str,
        *,
        idempotency_key: str,
        source_operation_id: str | None = None,
    ) -> MonitorProposal:
        async with self._mutation_lock:
            self._require_open()
            return await self._monitor_service.propose_natural(
                monitor_id,
                request,
                idempotency_key=idempotency_key,
                source_operation_id=source_operation_id,
            )

    async def confirm_monitor(
        self,
        proposal_id: str,
        *,
        candidate_hash: str,
        actor_id: str,
        reason: str,
    ) -> MonitorInspection:
        async with self._mutation_lock:
            self._require_open()
            return await self._monitor_service.confirm(
                proposal_id,
                candidate_hash=candidate_hash,
                actor_id=actor_id,
                reason=reason,
            )

    async def reject_monitor(
        self,
        proposal_id: str,
        *,
        candidate_hash: str,
        actor_id: str,
        reason: str,
    ) -> MonitorConfirmation:
        async with self._mutation_lock:
            self._require_open()
            return await self._monitor_service.reject(
                proposal_id,
                candidate_hash=candidate_hash,
                actor_id=actor_id,
                reason=reason,
            )

    async def list_monitors(
        self,
        *,
        statuses: tuple[MonitorStatus, ...] | None = None,
        include_deleted: bool = False,
        limit: int = 100,
    ) -> tuple[Monitor, ...]:
        self._require_open()
        return await self._monitor_service.list(
            statuses=statuses,
            include_deleted=include_deleted,
            limit=limit,
        )

    async def list_monitor_proposals(
        self,
        *,
        limit: int = 100,
    ) -> tuple[MonitorProposal, ...]:
        self._require_open()
        return await self._monitor_service.list_proposals(limit=limit)

    async def inspect_monitor(self, monitor_id: str) -> MonitorInspection:
        self._require_open()
        return await self._monitor_service.inspect(monitor_id)

    async def pause_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        async with self._mutation_lock:
            self._require_open()
            return await self._monitor_service.pause(
                monitor_id,
                actor_id=actor_id,
                reason=reason,
                idempotency_key=idempotency_key,
                operation_id=operation_id,
            )

    async def resume_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        async with self._mutation_lock:
            self._require_open()
            return await self._monitor_service.resume(
                monitor_id,
                actor_id=actor_id,
                reason=reason,
                idempotency_key=idempotency_key,
                operation_id=operation_id,
            )

    async def delete_monitor(
        self,
        monitor_id: str,
        *,
        actor_id: str,
        reason: str,
        idempotency_key: str,
        operation_id: str | None = None,
    ) -> MonitorInspection:
        async with self._mutation_lock:
            self._require_open()
            return await self._monitor_service.delete(
                monitor_id,
                actor_id=actor_id,
                reason=reason,
                idempotency_key=idempotency_key,
                operation_id=operation_id,
            )

    async def claim_monitor_run_now(
        self,
        monitor_id: str,
        *,
        idempotency_key: str,
        holder_id: str,
        lease_seconds: float = 300.0,
    ) -> MonitorClaimResult:
        async with self._mutation_lock:
            self._require_open()
            return await self._monitor_service.run_now(
                monitor_id,
                idempotency_key=idempotency_key,
                holder_id=holder_id,
                lease_seconds=lease_seconds,
            )

    async def transcript(self, session_id: str) -> SessionTranscript:
        self._require_open()
        transcript = await self._store.load_session(self.identity.id, session_id)
        if transcript is None:
            raise KeyError(f"unknown session: {session_id}")
        return transcript

    async def recall_memory(
        self,
        request: MemoryRecallRequest,
    ) -> MemoryRecallResult:
        self._require_open()
        if not isinstance(request, MemoryRecallRequest):
            raise TypeError("request must be a MemoryRecallRequest")
        self._require_memory_agent(request.scope.agent_id)
        return await self._memory_service.recall(request)

    async def list_memories(
        self,
        request: MemoryListRequest,
    ) -> MemoryListResult:
        self._require_open()
        if not isinstance(request, MemoryListRequest):
            raise TypeError("request must be a MemoryListRequest")
        self._require_memory_agent(request.scope.agent_id)
        return await self._memory_service.list(request)

    async def inspect_memory(
        self,
        request: MemoryInspectionRequest,
    ) -> MemoryInspection:
        self._require_open()
        if not isinstance(request, MemoryInspectionRequest):
            raise TypeError("request must be a MemoryInspectionRequest")
        self._require_memory_agent(request.agent_id)
        return await self._memory_service.inspect(request)

    async def supersede_memory(
        self,
        request: MemorySupersessionRequest,
    ) -> MemoryInspection:
        if not isinstance(request, MemorySupersessionRequest):
            raise TypeError("request must be a MemorySupersessionRequest")
        async with self._mutation_lock:
            self._require_open()
            self._require_memory_agent(request.agent_id)
            return await self._memory_service.supersede(request)

    async def restore_memory(
        self,
        request: MemoryRestoreRequest,
    ) -> MemoryInspection:
        if not isinstance(request, MemoryRestoreRequest):
            raise TypeError("request must be a MemoryRestoreRequest")
        async with self._mutation_lock:
            self._require_open()
            self._require_memory_agent(request.agent_id)
            return await self._memory_service.restore(request)

    async def refresh_skills(self) -> tuple[SkillIndex, ...]:
        async with self._mutation_lock:
            self._require_open()
            return await self._skill_service.refresh()

    async def list_skills(self) -> tuple[SkillIndex, ...]:
        self._require_open()
        return await self._skill_service.list()

    async def list_learning_proposals(
        self,
        *,
        operation_id: str | None = None,
        states: tuple[LearningProposalState, ...] = (
            LearningProposalState.PROPOSED,
            LearningProposalState.COMMITTED,
            LearningProposalState.REJECTED,
        ),
        limit: int = 100,
    ) -> tuple[LearningProposal, ...]:
        """List visible durable learning decisions in this agent's scope."""

        self._require_open()
        return await self._store.list_proposals(
            self.identity.id,
            operation_id=operation_id,
            states=states,
            limit=limit,
        )

    async def inspect_skill(self, skill_id: str) -> SkillInspection:
        self._require_open()
        return await self._skill_service.inspect(skill_id)

    async def activate_skill(
        self,
        skill_id: str,
        version_id: str,
        *,
        expected_active_version_id: str | None,
        actor_id: str,
        reason: str,
    ) -> SkillInspection:
        async with self._mutation_lock:
            self._require_open()
            return await self._skill_service.activate(
                skill_id,
                version_id,
                expected_active_version_id=expected_active_version_id,
                actor_id=actor_id,
                reason=reason,
            )

    async def propose_skill_change(
        self,
        source_operation_id: str,
        candidate: SkillChangeCandidate,
    ) -> SkillChangeProposalResult:
        if not isinstance(source_operation_id, str) or not source_operation_id.strip():
            raise ValueError("source_operation_id must be a non-empty string")
        if not isinstance(candidate, SkillChangeCandidate):
            raise TypeError("candidate must be a SkillChangeCandidate")
        async with self._mutation_lock:
            self._require_open()
            snapshot = (await self._store.load(source_operation_id)).snapshot
            if (
                snapshot.operation.agent_id != self.identity.id
                or snapshot.operation.status is not OperationStatus.SUCCEEDED
                or snapshot.loop_state.phase is not LoopPhase.TERMINAL
                or snapshot.trigger.kind is not TriggerKind.USER
            ):
                raise ValueError(
                    "skill-change source must be a completed successful user operation"
                )
            message = snapshot.trigger.payload.get("message")
            if not isinstance(message, str) or not message.strip():
                raise ValueError(
                    "skill-change source user operation must contain a message"
                )
            source_hash = (
                "sha256:"
                + sha256(
                    canonical_json(snapshot.trigger.payload).encode("utf-8")
                ).hexdigest()
            )
            return await self._skill_change_service.propose(
                candidate,
                LearningProvenance(
                    agent_id=self.identity.id,
                    operation_id=snapshot.operation.id,
                    trigger_id=snapshot.trigger.id,
                    source_outcome=LearningSourceOutcome.SUCCEEDED,
                    source_hash=source_hash,
                ),
            )

    async def accept_skill_change(
        self,
        proposal_id: str,
        *,
        expected_active_version_id: str | None,
        actor_id: str,
        reason: str,
    ) -> SkillChangeAcceptanceResult:
        async with self._mutation_lock:
            self._require_open()
            return await self._skill_change_service.accept(
                proposal_id,
                expected_active_version_id=expected_active_version_id,
                actor_id=actor_id,
                reason=reason,
            )

    async def learn_correction(
        self,
        operation_id: str,
    ) -> ExplicitCorrectionResult:
        async with self._mutation_lock:
            self._require_open()
            snapshot = (await self._store.load(operation_id)).snapshot
            return await self._learning_service.learn(snapshot)

    async def _learn_completed_operation(self, result: LoopExit) -> tuple[str, ...]:
        if result.kind is not LoopExitKind.COMPLETED:
            return ()
        try:
            snapshot = (await self._store.load(result.operation_id)).snapshot
        except Exception:
            return ("learning.post_operation_unavailable",)
        message = snapshot.trigger.payload.get("message")
        if is_explicit_learning_message(message):
            try:
                await self._learning_service.learn(snapshot)
            except ExplicitCorrectionLearningError:
                return ("learning.correction_failed",)
            except Exception:
                return ("learning.post_operation_unavailable",)
            return ()

        notices: list[str] = []
        try:
            fact = await self._learning_service.propose_evidence_fact(snapshot)
            if fact is not None:
                notices.append(f"learning.fact_{fact.state.value}")
        except Exception:
            notices.append("learning.post_operation_unavailable")
        try:
            if (
                not isinstance(message, str)
                or snapshot.trigger.kind is not TriggerKind.USER
            ):
                skill = None
            else:
                source_hash = (
                    "sha256:"
                    + sha256(
                        canonical_json(snapshot.trigger.payload).encode("utf-8")
                    ).hexdigest()
                )
                skill = await self._skill_change_service.propose_natural(
                    message,
                    LearningProvenance(
                        agent_id=self.identity.id,
                        operation_id=snapshot.operation.id,
                        trigger_id=snapshot.trigger.id,
                        source_outcome=LearningSourceOutcome.SUCCEEDED,
                        source_hash=source_hash,
                    ),
                )
            if skill is not None:
                notices.append(f"learning.skill_{skill.proposal.state.value}")
        except Exception:
            if "learning.post_operation_unavailable" not in notices:
                notices.append("learning.post_operation_unavailable")
        return tuple(notices)

    async def _with_post_operation_learning(self, result: LoopExit) -> LoopExit:
        notices = await self._learn_completed_operation(result)
        return (
            result
            if not notices
            else replace(
                result,
                post_operation_notices=(*result.post_operation_notices, *notices),
            )
        )

    def _require_memory_agent(self, agent_id: str) -> None:
        if agent_id != self.identity.id:
            raise ValueError("memory request belongs to another agent")

    async def close(self) -> None:
        async with self._mutation_lock:
            if self._closed:
                return
            try:
                await self._store.close()
            finally:
                self._closed = True
                self._writer_lock.release()

    async def _ensure_session(self, session_id: str, message: str) -> Session:
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        existing = await self._store.load_session(self.identity.id, session_id)
        if existing is not None:
            await self._require_session_idle(session_id)
            return existing.session
        now = self._clock()
        session = Session(
            id=session_id,
            agent_id=self.identity.id,
            title=message.strip()[:80],
            created_at=now,
            updated_at=now,
        )
        try:
            return await self._store.create_session(session)
        except SessionAlreadyExistsError:
            raced = await self._store.load_session(self.identity.id, session_id)
            if raced is None:
                raise
            await self._require_session_idle(session_id)
            return raced.session

    async def _require_session_idle(self, session_id: str) -> None:
        nonterminal = await self._store.load_nonterminal(self.identity.id)
        active = tuple(
            item.snapshot.operation
            for item in nonterminal
            if item.snapshot.operation.session_id == session_id
        )
        if not active:
            return
        operation = active[0]
        raise SessionOperationActiveError(
            session_id,
            operation.id,
            operation.status.value,
        )

    def _require_loop(self) -> AgentLoop:
        self._require_open()
        if self._loop is None:
            raise AgentNotConfiguredError(
                "agent execution requires model, context_builder, and domain"
            )
        return self._loop

    def _require_open(self) -> None:
        if self._closed:
            raise AgentHomeError("embedded agent is closed")


def _admit_agent_home(
    name: str,
    root: str | Path | None,
    create: bool,
) -> tuple[Path, _WriterLock]:
    if not isinstance(name, str) or _AGENT_NAME.fullmatch(name) is None:
        raise AgentNameError(
            "agent name must contain 1-64 ASCII letters, digits, '_' or '-'"
        )
    state_root = Path.home() / ".daita-next" if root is None else Path(root)
    if ".." in state_root.parts:
        raise AgentHomeError("agent state root cannot contain parent aliases")
    state_root = Path(os.path.abspath(os.fspath(state_root)))
    resolved_root = _require_unaliased_path(state_root, "agent state root")
    v1_root = (Path.home() / ".daita").resolve(strict=False)
    if resolved_root == v1_root or resolved_root.is_relative_to(v1_root):
        raise AgentHomeError("v2 refuses to use paths under the v1 ~/.daita root")
    state_root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _require_unaliased_path(state_root, "agent state root")
    agents_root = state_root / "agents"
    _require_unaliased_path(agents_root, "agents directory")
    agents_root.mkdir(mode=0o700, exist_ok=True)
    _require_unaliased_path(agents_root, "agents directory")
    os.chmod(agents_root, 0o700)
    home = agents_root / name
    _require_unaliased_path(home, "agent home")
    if not create and not home.is_dir():
        raise AgentNotFoundError(f"agent does not exist: {name}")
    home.mkdir(mode=0o700, exist_ok=True)
    _require_unaliased_path(home, "agent home")
    os.chmod(home, 0o700)
    run = home / "run"
    _require_unaliased_path(run, "agent run directory")
    run.mkdir(mode=0o700, exist_ok=True)
    _require_unaliased_path(run, "agent run directory")
    os.chmod(run, 0o700)
    return home, _WriterLock.acquire(run / "host.lock")


def _ensure_agent_directory(home: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "/" in name or name in {".", ".."}:
        raise AgentHomeError("agent service directory name is invalid")
    path = home / name
    _require_unaliased_path(path, f"agent {name} directory")
    try:
        path.mkdir(mode=0o700, exist_ok=True)
    except OSError as error:
        raise AgentHomeError(f"cannot create agent {name} directory") from error
    resolved = _require_unaliased_path(path, f"agent {name} directory")
    if not resolved.is_dir():
        raise AgentHomeError(f"agent {name} path must be a directory")
    os.chmod(resolved, 0o700)
    return resolved


def _require_unaliased_path(path: Path, label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(path)))
    resolved = lexical.resolve(strict=False)
    if lexical != resolved or lexical.is_symlink():
        raise AgentHomeError(f"{label} cannot contain a symlink or path alias")
    return resolved


def _write_manifest(home: Path, identity: AgentIdentity) -> None:
    temporary_name = f".agent.toml.{uuid4().hex}.tmp"
    text = (
        "manifest_version = 1\n"
        f'agent_id = "{identity.id}"\n'
        f'display_name = "{identity.display_name}"\n'
        'state_path = "state.db"\n'
        f"state_schema_generation = {identity.state_schema_generation}\n"
        f'created_at = "{identity.created_at.isoformat()}"\n'
    )
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        directory_flags |= os.O_NOFOLLOW
    directory = os.open(home, directory_flags)
    try:
        file_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_NOFOLLOW"):
            file_flags |= os.O_NOFOLLOW
        descriptor = os.open(temporary_name, file_flags, 0o600, dir_fd=directory)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as file:
                file.write(text)
                file.flush()
                os.fsync(file.fileno())
            os.replace(
                temporary_name,
                "agent.toml",
                src_dir_fd=directory,
                dst_dir_fd=directory,
            )
            os.fsync(directory)
        finally:
            try:
                os.unlink(temporary_name, dir_fd=directory)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory)


def _read_manifest(home: Path, expected_name: str) -> AgentIdentity:
    path = home / "agent.toml"
    state_path = home / "state.db"
    if path.is_symlink() or state_path.is_symlink():
        raise AgentIdentityMismatchError("agent bootstrap files cannot be symlinks")
    if not path.is_file() or not state_path.is_file():
        raise AgentNotFoundError("agent home is missing agent.toml or state.db")
    try:
        _verify_regular_no_follow(state_path, "agent state database")
        flags = os.O_RDONLY
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "r", encoding="utf-8") as file:
            values = tomllib.loads(file.read())
        if set(values) != {
            "manifest_version",
            "agent_id",
            "display_name",
            "state_path",
            "state_schema_generation",
            "created_at",
        }:
            raise ValueError("unexpected manifest fields")
        if values["manifest_version"] != 1 or values["state_path"] != "state.db":
            raise ValueError("unsupported manifest version or state path")
        if values["display_name"] != expected_name:
            raise ValueError("manifest display name does not match its home")
        created_at = datetime.fromisoformat(values["created_at"])
        return AgentIdentity(
            id=values["agent_id"],
            display_name=values["display_name"],
            created_at=created_at,
            state_schema_generation=values["state_schema_generation"],
        )
    except (OSError, tomllib.TOMLDecodeError, KeyError, TypeError, ValueError) as error:
        raise AgentIdentityMismatchError("agent.toml is invalid") from error


def _verify_regular_no_follow(path: Path, label: str) -> None:
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise AgentIdentityMismatchError(f"{label} must be a regular file")
    finally:
        os.close(descriptor)


def _cleanup_failed_create(home: Path) -> None:
    """Remove only bootstrap files that this locked create could have made."""

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
