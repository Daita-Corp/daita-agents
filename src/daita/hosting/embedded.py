"""Small in-process composition for one persistent data agent."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
import errno
import os
from pathlib import Path
import re
import stat
import tomllib
from typing import Self, TypeVar
from uuid import uuid4

from .._json import FrozenJsonObject
from ..adapters.local_files import LocalDirectoryReadBackend
from ..adapters.models import DiscoveryRequest, SourceRegistration
from ..adapters.postgresql_query import PostgreSQLQueryBackend
from ..adapters.protocols import ResourceAdapter, ResourceAdapterError, ResourceSource
from ..adapters.sqlite_query import SQLiteQueryBackend
from ..catalog.capabilities import catalog_declarations
from ..catalog.models import (
    CatalogResource,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSync,
    CatalogSyncStatus,
)
from ..catalog.service import CatalogService
from ..capabilities import ApprovalHandler, CapabilityRegistry
from ..config import AgentConfig
from ..domains.data import (
    CatalogDataView,
    DataContextBuilder,
    DataToolRuntime,
    local_file_read_declarations,
    postgresql_query_declarations,
    sqlite_query_declarations,
)
from ..domains.data.context import _project_completed_history
from ..errors import AgentError
from ..identity import AgentIdentity
from ..llm.factory import create_model_route_provider
from ..llm.models import ModelProfile
from ..llm.protocols import ModelProvider
from ..llm.routing import ModelRoute
from ..loop.driver import (
    AgentLoop,
    ContextBuilder,
    ToolRuntime,
)
from ..loop.models import ConversationRun, LoopExit, LoopLimits, RunInput, Transcript
from ..memory import MemoryStore
from ..memory.capabilities import memory_set_declarations
from ..observation import AgentObserver
from ..security import SecretProvider
from ..skills import Skill, SkillStore, SkillSummary
from ..skills.capabilities import skill_declarations
from ..storage.sqlite import SQLiteStateStore

_AGENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_CONVERSATION_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_T = TypeVar("_T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _validate_conversation_id(value: str) -> None:
    if not isinstance(value, str) or _CONVERSATION_ID.fullmatch(value) is None:
        raise ValueError("conversation_id must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


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


class HostActiveError(AgentHomeError):
    code = "host_active"


class AgentNotConfiguredError(AgentHomeError):
    pass


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
        memory_store: MemoryStore,
        skill_store: SkillStore,
        mutation_lock: asyncio.Lock,
        model_profile: ModelProfile | None,
        model_route: ModelRoute | None,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        self.identity = identity
        self.home = home
        self.model_profile = model_profile
        self.model_route = model_route
        self._writer_lock = writer_lock
        self._store = store
        self._loop = loop
        self._transcripts = transcripts
        self._capabilities = capabilities
        self._catalog_service = catalog_service
        self._memory_store = memory_store
        self._skill_store = skill_store
        self._clock = clock
        self._id_factory = id_factory
        self._mutation_lock = mutation_lock
        self._run_lock = asyncio.Lock()
        self._closed = False

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
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> Self:
        resolved_clock = clock or _utc_now
        resolved_ids = id_factory or _new_id
        model, model_profile, model_route, limits = _resolve_configuration(
            config,
            model=model,
            model_profile=model_profile,
            limits=limits,
            secret_provider=secret_provider,
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
            embedded = cls._compose(
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
                secret_provider=secret_provider,
                observer=observer,
                approval_handler=approval_handler,
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
        observer: AgentObserver | None = None,
        approval_handler: ApprovalHandler | None = None,
    ) -> Self:
        resolved_clock = clock or _utc_now
        resolved_ids = id_factory or _new_id
        model, model_profile, model_route, limits = _resolve_configuration(
            config,
            model=model,
            model_profile=model_profile,
            limits=limits,
            secret_provider=secret_provider,
        )
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
            return cls._compose(
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
                secret_provider=secret_provider,
                observer=observer,
                approval_handler=approval_handler,
            )
        except BaseException:
            if store is not None:
                await store.close()
            writer_lock.release()
            raise

    @classmethod
    def _compose(
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
        secret_provider: SecretProvider | None,
        observer: AgentObserver | None,
        approval_handler: ApprovalHandler | None,
    ) -> Self:
        catalog_service = CatalogService(store, store)
        data_view = CatalogDataView(store, catalog_service, store)
        catalog = catalog_declarations(identity.id, catalog_service)
        sqlite = sqlite_query_declarations(
            identity.id, SQLiteQueryBackend(store, data_view)
        )
        postgresql = postgresql_query_declarations(
            identity.id,
            PostgreSQLQueryBackend(store, data_view, secret_provider),
        )
        local_files = local_file_read_declarations(
            identity.id, LocalDirectoryReadBackend(store, store)
        )
        mutation_lock = asyncio.Lock()
        memory_store = MemoryStore(home, mutation_lock)
        skill_store = SkillStore(home, mutation_lock)
        memory = memory_set_declarations(memory_store)
        skills = skill_declarations(skill_store)
        capabilities = CapabilityRegistry(
            capabilities=(
                *catalog.capabilities,
                *sqlite.capabilities,
                *postgresql.capabilities,
                *local_files.capabilities,
                *memory.capabilities,
                *skills.capabilities,
            ),
            executors=(
                *catalog.executors,
                *sqlite.executors,
                *postgresql.executors,
                *local_files.executors,
                *memory.executors,
                *skills.executors,
            ),
            tool_views=(
                *catalog.tool_views,
                *sqlite.tool_views,
                *postgresql.tool_views,
                *local_files.tool_views,
                *memory.tool_views,
                *skills.tool_views,
            ),
        )
        resolved_context = context_builder
        resolved_tools = tools
        if model is not None and resolved_context is None:
            assert model_profile is not None
            if not model_profile.supports_tools:
                raise AgentNotConfiguredError(
                    "the data agent requires a tool-capable model profile"
                )
            resolved_context = DataContextBuilder(
                data_view,
                profile=model_profile,
                memory=memory_store,
                skills=skill_store,
            )
            resolved_tools = DataToolRuntime(
                capabilities,
                data_view,
                approval_handler=approval_handler,
                mutation_lock=mutation_lock,
                observer=observer,
                clock=clock,
            )
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
            )
        )
        return cls(
            identity=identity,
            home=home,
            writer_lock=writer_lock,
            store=store,
            loop=loop,
            transcripts=transcripts,
            capabilities=capabilities,
            catalog_service=catalog_service,
            memory_store=memory_store,
            skill_store=skill_store,
            mutation_lock=mutation_lock,
            model_profile=model_profile,
            model_route=model_route,
            clock=clock,
            id_factory=id_factory,
        )

    async def run(
        self,
        message: str,
        *,
        conversation_id: str | None = None,
    ) -> LoopExit:
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        supplied_conversation = conversation_id is not None
        resolved_conversation = (
            self._id_factory("conversation")
            if conversation_id is None
            else conversation_id
        )
        _validate_conversation_id(resolved_conversation)
        loop = self._require_loop()
        async with self._run_lock:
            self._require_open()
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
            prior_messages = _project_completed_history(
                conversation,
                older_history_exists=older_history_exists,
            )
            return await loop.run(
                RunInput(
                    id=self._id_factory("run"),
                    agent_id=self.identity.id,
                    message=message.strip(),
                    created_at=self._clock(),
                    conversation_id=resolved_conversation,
                ),
                prior_messages=prior_messages,
            )

    async def transcript(self, run_id: str) -> Transcript:
        self._require_open()
        return await self._transcripts.load(run_id)

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

    async def attach(self, source: ResourceSource) -> SourceRegistration:
        async with self._mutation_lock:
            self._require_open()
            adapter = await source.open(
                agent_id=self.identity.id,
                attached_at=self._clock(),
                clock=self._clock,
            )
            if not isinstance(adapter, ResourceAdapter):
                raise TypeError("source open() must return ResourceAdapter")
            registration = adapter.registration
            sync: CatalogSync | None = None
            try:
                self._capabilities.validate_declarations(adapter.declarations())
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
                existing = await self._store.load_source(
                    self.identity.id, registration.id
                )
                if existing is None:
                    await self._store.register_source(registration)
                elif existing != registration:
                    raise AgentHomeError(
                        f"source registration already exists: {registration.id}"
                    )
                await self._store.commit_snapshot(discovery.snapshot)
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

    async def detach(self, source_id: str) -> SourceRegistration:
        async with self._mutation_lock:
            self._require_open()
            return await self._store.detach_source(
                self.identity.id, source_id, self._clock()
            )

    async def list_sources(self) -> tuple[SourceRegistration, ...]:
        self._require_open()
        return await self._store.list_sources(self.identity.id)

    async def list_catalog_resources(
        self, *, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        self._require_open()
        return await self._store.list_resources(self.identity.id, source_id)

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
        if self._closed:
            return
        self._closed = True
        try:
            await self._memory_store.close()
            await self._skill_store.close()
            await self._store.close()
        finally:
            self._writer_lock.release()

    def _require_loop(self) -> AgentLoop:
        self._require_open()
        if self._loop is None:
            raise AgentNotConfiguredError("agent execution requires a model")
        return self._loop

    def _require_open(self) -> None:
        if self._closed:
            raise AgentHomeError("embedded agent is closed")


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


def _admit_agent_home(
    name: str,
    root: str | Path | None,
    create: bool,
) -> tuple[Path, _WriterLock]:
    if not isinstance(name, str) or _AGENT_NAME.fullmatch(name) is None:
        raise AgentNameError(
            "agent name must contain 1-64 ASCII letters, digits, '_' or '-'"
        )
    state_root = Path.home() / ".daita" if root is None else Path(root)
    if ".." in state_root.parts:
        raise AgentHomeError("agent state root cannot contain parent aliases")
    state_root = Path(os.path.abspath(os.fspath(state_root)))
    state_root = _require_unaliased_path(state_root, "agent state root")
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
        (agents_root, "agents directory"),
        (home, "agent home"),
        (run, "agent run directory"),
    ):
        _require_unaliased_path(path, label)
        os.chmod(path, 0o700)
    return home, _WriterLock.acquire(run / "host.lock")


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


__all__ = [
    "AgentAlreadyExistsError",
    "AgentHomeError",
    "AgentIdentityMismatchError",
    "AgentNameError",
    "AgentNotConfiguredError",
    "AgentNotFoundError",
    "EmbeddedAgent",
    "HostActiveError",
]
