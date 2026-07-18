"""Embedded composition for one isolated persistent agent home."""

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

from ..capabilities import CapabilityRegistry
from ..identity import AgentIdentity
from ..llm.protocols import ModelProvider
from ..loop.driver import AgentLoop, ContextBuilder, DomainController
from ..loop.models import LoopBudgets, LoopExit
from ..operations.checkpoints import OperationSnapshot
from ..operations.governance import DefaultPolicyEvaluator
from ..operations.models import AgentTrigger, TriggerKind
from ..operations.runtime import OperationRuntime
from ..sessions import Session, SessionAlreadyExistsError, SessionTranscript
from ..storage.blobs import LocalBlobStore
from ..storage.sqlite import SQLiteOperationStore

_AGENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")
_T = TypeVar("_T")


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _validate_loop_configuration(
    model: ModelProvider | None,
    context_builder: ContextBuilder | None,
    domain: DomainController | None,
) -> None:
    configured = (
        model is not None,
        context_builder is not None,
        domain is not None,
    )
    if any(configured) and not all(configured):
        raise AgentNotConfiguredError(
            "model, context_builder, and domain must be configured together"
        )


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


class AgentHomeError(RuntimeError):
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
        loop: AgentLoop | None,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> None:
        self.identity = identity
        self.home = home
        self._writer_lock = writer_lock
        self._store = store
        self._loop = loop
        self._clock = clock
        self._id_factory = id_factory
        self._mutation_lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def create(
        cls,
        name: str,
        *,
        root: str | Path | None = None,
        model: ModelProvider | None = None,
        context_builder: ContextBuilder | None = None,
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets = LoopBudgets(),
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> Self:
        _validate_loop_configuration(model, context_builder, domain)
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
            created_at = resolved_clock()
            identity = AgentIdentity(
                id=resolved_id_factory("agent"),
                display_name=name,
                created_at=created_at,
            )
            store = await SQLiteOperationStore.open(state_path)
            await store.initialize_identity(identity)
            _, manifest_cancelled = await _await_sync_completion(
                lambda: _write_manifest(home, identity)
            )
            bootstrap_published = True
            if manifest_cancelled:
                raise asyncio.CancelledError
            return cls._compose(
                identity=identity,
                home=home,
                writer_lock=writer_lock,
                store=store,
                model=model,
                context_builder=context_builder,
                domain=domain,
                capabilities=capabilities,
                policy=policy,
                budgets=budgets,
                clock=resolved_clock,
                id_factory=resolved_id_factory,
            )
        except BaseException:
            try:
                if store is not None:
                    await store.close()
            finally:
                try:
                    if not bootstrap_published:
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
        model: ModelProvider | None = None,
        context_builder: ContextBuilder | None = None,
        domain: DomainController | None = None,
        capabilities: CapabilityRegistry | None = None,
        policy: DefaultPolicyEvaluator | None = None,
        budgets: LoopBudgets = LoopBudgets(),
        clock: Callable[[], datetime] | None = None,
        id_factory: Callable[[str], str] | None = None,
    ) -> Self:
        _validate_loop_configuration(model, context_builder, domain)
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
            store = await SQLiteOperationStore.open(home / "state.db")
            identity = await store.load_identity()
            if identity is None or identity != manifest:
                raise AgentIdentityMismatchError(
                    "agent.toml does not match authoritative state.db identity"
                )
            return cls._compose(
                identity=identity,
                home=home,
                writer_lock=writer_lock,
                store=store,
                model=model,
                context_builder=context_builder,
                domain=domain,
                capabilities=capabilities,
                policy=policy,
                budgets=budgets,
                clock=resolved_clock,
                id_factory=resolved_id_factory,
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
        context_builder: ContextBuilder | None,
        domain: DomainController | None,
        capabilities: CapabilityRegistry | None,
        policy: DefaultPolicyEvaluator | None,
        budgets: LoopBudgets,
        clock: Callable[[], datetime],
        id_factory: Callable[[str], str],
    ) -> Self:
        _validate_loop_configuration(model, context_builder, domain)
        runtime = OperationRuntime(
            clock=clock,
            id_factory=id_factory,
            capabilities=capabilities,
            store=store,
            blob_store=LocalBlobStore(home / "blobs"),
            policy=policy,
        )
        loop = (
            None
            if model is None or context_builder is None or domain is None
            else AgentLoop(
                runtime=runtime,
                model=model,
                context_builder=context_builder,
                domain=domain,
                budgets=budgets,
            )
        )
        return cls(
            identity=identity,
            home=home,
            writer_lock=writer_lock,
            store=store,
            loop=loop,
            clock=clock,
            id_factory=id_factory,
        )

    async def run(self, message: str, *, session_id: str | None = None) -> LoopExit:
        loop = self._require_loop()
        if not isinstance(message, str) or not message.strip():
            raise ValueError("message must be a non-empty string")
        async with self._mutation_lock:
            self._require_open()
            if session_id is not None:
                await self._ensure_session(session_id, message)
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
            return await loop.run(trigger)

    async def inspect(self, operation_id: str) -> OperationSnapshot:
        self._require_open()
        return (await self._store.load(operation_id)).snapshot

    async def resume(self, operation_id: str) -> LoopExit:
        loop = self._require_loop()
        async with self._mutation_lock:
            self._require_open()
            return await loop.resume(operation_id)

    async def transcript(self, session_id: str) -> SessionTranscript:
        self._require_open()
        transcript = await self._store.load_session(self.identity.id, session_id)
        if transcript is None:
            raise KeyError(f"unknown session: {session_id}")
        return transcript

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
            return raced.session

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
