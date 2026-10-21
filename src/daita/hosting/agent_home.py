"""Admit, discover, validate and delete local agent-home directories."""

from __future__ import annotations

import errno
import os
import re
import shutil
import stat
import tomllib
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from ..adapters.local_workspace import LocalWorkspaceBackend
from ..artifacts.delivery import validate_delivery_configuration
from ..artifacts.store import validate_artifact_home
from ..errors import StateCompatibilityCode, StateCompatibilityError
from ..identity import AgentIdentity
from ..memory.store import validate_memory_documents
from ..skills.store import validate_skill_documents
from ..storage.sqlite import (
    SQLiteStateStore,
    load_current_artifact_inventory,
    validate_current_state_database,
)
from ..workspace import LocalFileAccess, LocalWorkspace
from .configuration import (
    _MODEL_CONFIG_NAME,
    _read_model_configuration,
    _read_model_configuration_document,
)
from .errors import (
    AgentHomeError,
    AgentIdentityMismatchError,
    AgentNameError,
    AgentNotFoundError,
    HostActiveError,
)

_AGENT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}\Z")


_MAX_AGENT_HOME_CANDIDATES = 256


# Who holds an agent's writer lock: "host", "tui-inprocess" or "script".
WRITER_ROLE: ContextVar[str] = ContextVar("daita_writer_role", default="script")
_HOLDER_LABELS = {"tui-inprocess": "the Daita TUI", "script": "a Python script"}


_LEGACY_STATE_ROOT_MARKERS = (
    "catalog.json",
    "graph",
    "memory",
    "sessions",
)


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
                    holder = read_writer_holder(os.pread(descriptor, 4096, 0))
                    raise HostActiveError(
                        _holder_message(path.parent.parent.name, holder),
                        holder=holder,
                    ) from error
                raise
            started_at = datetime.now(UTC).replace(microsecond=0).isoformat()
            os.ftruncate(descriptor, 0)
            os.write(
                descriptor,
                f"pid={os.getpid()}\nrole={WRITER_ROLE.get()}\n"
                f"started_at={started_at.replace('+00:00', 'Z')}\n".encode("ascii"),
            )
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


def read_writer_holder(content: bytes) -> dict[str, object]:
    """Parse writer-lock contents, including the older ``pid=``-only format."""

    holder: dict[str, object] = {"role": None, "pid": None, "started_at": None}
    for line in content.decode("ascii", "replace").splitlines():
        key, _, value = line.partition("=")
        if key == "pid" and value.isdigit():
            holder["pid"] = int(value)
        elif key in {"role", "started_at"} and value:
            holder[key] = value
    return holder


def _holder_message(name: str, holder: dict[str, object]) -> str:
    pid = "unknown" if holder["pid"] is None else holder["pid"]
    if holder["role"] == "host":
        return (
            f'Agent "{name}" is run by the Daita host (pid {pid}).\n'
            f'Use daita.connect("{name}") from a script, or run `daita host stop`\n'
            "to use it directly in this process."
        )
    label = _HOLDER_LABELS.get(str(holder["role"]), "another Daita process")
    return f'Agent "{name}" is in use by {label} (pid {pid}).'


def describe_writer_holder(holder: Mapping[str, object]) -> str:
    """Say who holds a writer lock: "In use by a Python script (pid 5530)."."""

    pid = holder.get("pid") or "unknown"
    label = _HOLDER_LABELS.get(str(holder.get("role")), "another Daita process")
    return f"In use by {label} (pid {pid})."


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


def _validate_workspace_composition(
    workspace: LocalWorkspace | None,
    *,
    hosted: bool,
) -> None:
    if not isinstance(hosted, bool):
        raise TypeError("hosted must be bool")
    if hosted:
        if workspace is not None:
            raise ValueError("hosted composition cannot admit local file access")
        return
    if not isinstance(workspace, LocalWorkspace):
        raise TypeError("local agent composition requires LocalWorkspace")


async def _open_workspace_backend(
    workspace: LocalWorkspace | None,
    *,
    hosted: bool,
    home: Path,
    clock: Callable[[], datetime],
) -> LocalWorkspaceBackend | None:
    if hosted:
        return None
    assert workspace is not None
    if workspace.access is LocalFileAccess.NONE:
        return None
    return await LocalWorkspaceBackend.open(
        workspace,
        agent_root=home.parent.parent,
        agent_home=home,
        clock=clock,
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


def _validate_agent_home_target(
    manifest: AgentIdentity,
    source_home: Path,
    candidate_home: Path,
    affected_paths: frozenset[str],
) -> None:
    """Validate one complete home view assembled by the upgrade coordinator."""

    state_home = (
        candidate_home
        if candidate_home == source_home or "state.db" in affected_paths
        else source_home
    )
    identity = validate_current_state_database(state_home / "state.db")
    if identity is None or identity != manifest:
        raise AgentIdentityMismatchError(
            "agent.toml does not match the upgraded state identity"
        )

    config_home = (
        candidate_home
        if candidate_home == source_home or _MODEL_CONFIG_NAME in affected_paths
        else source_home
    )
    _read_model_configuration_document(config_home)
    if candidate_home != source_home:
        _read_model_configuration(config_home, manifest.id)
    memory_home = (
        candidate_home
        if candidate_home == source_home
        or {"MEMORY.md", "USER.md"}.intersection(affected_paths)
        else source_home
    )
    validate_memory_documents(memory_home)
    validate_skill_documents(source_home)
    records = load_current_artifact_inventory(state_home / "state.db", manifest.id)
    validate_artifact_home(
        agent_id=manifest.id, agent_home=source_home, records=records
    )
    validate_delivery_configuration(source_home)


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
