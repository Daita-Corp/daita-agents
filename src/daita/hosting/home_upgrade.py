"""Coordinate the one crash-recoverable agent-home revision sequence."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote
from uuid import uuid4

from ..errors import StateCompatibilityCode, StateCompatibilityError
from ..storage.home_migrations import (
    CURRENT_HOME_REVISION,
    HOME_MIGRATIONS,
    MINIMUM_SUPPORTED_HOME_REVISION,
    HomeMigrationJournalError,
    HomeMigrationJournalNewerError,
    insert_migration_row,
    inspect_home_revision,
)
from ..storage.home_migrations.models import HomeMigration
from ..storage.home_migrations.revision_0001 import detect_preproduction_shape
from ..storage.schema_contract import require_healthy, require_schema

_UPGRADE_DIRECTORY = ".home-upgrade"
_ROLLBACK_DIRECTORY = ".home-rollbacks"
_JOURNAL_NAME = "journal.json"
_MAX_JOURNAL_BYTES = 256 * 1024
_DISK_HEADROOM_BYTES = 1024 * 1024

HomeValidator = Callable[[Path, Path, frozenset[str]], None]
PhaseHook = Callable[[str], None]


@dataclass(frozen=True, slots=True)
class AgentHomeStatus:
    current_revision: int
    found_revision: int | None
    minimum_supported_revision: int
    source_kind: str
    upgrade_required: bool
    recovery_required: bool


@dataclass(frozen=True, slots=True)
class AgentHomeUpgradeResult:
    source_revision: int
    target_revision: int
    upgraded: bool
    recovered: bool
    rollback_path: Path | None = None


def _connect_read_only(path: Path) -> sqlite3.Connection:
    uri = f"file:{quote(os.fspath(path))}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.execute("PRAGMA query_only = ON")
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _error(
    code: StateCompatibilityCode,
    home: Path,
    message: str,
    *,
    found: int | str | None,
) -> StateCompatibilityError:
    return StateCompatibilityError(
        code,
        home / "state.db",
        message,
        current_revision=str(CURRENT_HOME_REVISION),
        found_revision=None if found is None else str(found),
    )


def _require_regular(path: Path, *, optional: bool = False) -> None:
    try:
        facts = path.lstat()
    except FileNotFoundError:
        if optional:
            return
        raise ValueError(f"required agent-home file is missing: {path.name}") from None
    if not stat.S_ISREG(facts.st_mode) or path.is_symlink():
        raise ValueError(f"agent-home file is not a regular file: {path.name}")


def _contained_home_path(home: Path, relative: str) -> Path:
    """Resolve one registry-owned path without traversing an aliased parent."""

    candidate = home / relative
    current = home
    for part in Path(relative).parts[:-1]:
        current = current / part
        try:
            facts = current.lstat()
        except FileNotFoundError:
            break
        if not stat.S_ISDIR(facts.st_mode) or stat.S_ISLNK(facts.st_mode):
            raise ValueError(f"agent-home path parent is invalid: {relative}")
    try:
        candidate.relative_to(home)
    except ValueError:
        raise ValueError(f"agent-home path escapes its home: {relative}") from None
    return candidate


def _resolve_home(home: Path) -> Path:
    if not isinstance(home, Path) or not home.is_absolute() or ".." in home.parts:
        raise ValueError("agent home must be an absolute path without aliases")
    lexical = Path(os.path.abspath(os.fspath(home)))
    facts = lexical.lstat()
    resolved = lexical.resolve(strict=True)
    if (
        not stat.S_ISDIR(facts.st_mode)
        or stat.S_ISLNK(facts.st_mode)
        or resolved != lexical
    ):
        raise ValueError("agent home must be an exact non-symlink directory")
    return lexical


def _has_table(connection: sqlite3.Connection, name: str) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
            (name,),
        ).fetchone()
        is not None
    )


def inspect_agent_home(home: Path) -> AgentHomeStatus:
    """Return compatibility facts without changing the home."""

    home = _resolve_home(home)
    try:
        _require_regular(home / "agent.toml")
        _require_regular(home / "state.db")
        _require_regular(home / "config.json", optional=True)
        upgrade_path = home / _UPGRADE_DIRECTORY
        recovery_required = upgrade_path.exists() or upgrade_path.is_symlink()
        with _connect_read_only(home / "state.db") as connection:
            if _has_table(connection, "agent_home_migrations"):
                try:
                    revision = inspect_home_revision(connection)
                except HomeMigrationJournalNewerError as error:
                    raise _error(
                        StateCompatibilityCode.NEWER_REVISION,
                        home,
                        "This agent home was written by a newer Daita release. "
                        "No local data was changed.",
                        found=error.found_revision,
                    ) from None
                except HomeMigrationJournalError as error:
                    raise _error(
                        StateCompatibilityCode.REVISION_UNSUPPORTED,
                        home,
                        "This agent home has an unknown, incomplete, reordered, "
                        "or edited migration history. No local data was changed.",
                        found=error.found_revision,
                    ) from None
                if revision > CURRENT_HOME_REVISION:
                    raise _error(
                        StateCompatibilityCode.NEWER_REVISION,
                        home,
                        "This agent home was written by a newer Daita release. "
                        "No local data was changed.",
                        found=revision,
                    )
                if revision < MINIMUM_SUPPORTED_HOME_REVISION:
                    raise _error(
                        StateCompatibilityCode.REVISION_UNSUPPORTED,
                        home,
                        "This agent-home revision is outside this release's "
                        "documented support window. No local data was changed.",
                        found=revision,
                    )
                return AgentHomeStatus(
                    current_revision=CURRENT_HOME_REVISION,
                    found_revision=revision,
                    minimum_supported_revision=MINIMUM_SUPPORTED_HOME_REVISION,
                    source_kind="production",
                    upgrade_required=revision < CURRENT_HOME_REVISION,
                    recovery_required=recovery_required,
                )
            source_shape = detect_preproduction_shape(connection)
    except StateCompatibilityError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise _error(
            StateCompatibilityCode.DAMAGED,
            home,
            "This agent home or model configuration is damaged, or its "
            "preproduction shape is unsupported. No local data was changed.",
            found=None,
        ) from None
    return AgentHomeStatus(
        current_revision=CURRENT_HOME_REVISION,
        found_revision=None,
        minimum_supported_revision=MINIMUM_SUPPORTED_HOME_REVISION,
        source_kind=source_shape,
        upgrade_required=True,
        recovery_required=recovery_required,
    )


def _sha256(path: Path) -> str:
    _require_regular(path)
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, value: dict[str, object]) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > _MAX_JOURNAL_BYTES:
        raise ValueError("agent-home upgrade journal exceeds its bound")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    descriptor: int | None = None
    published = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        view = memoryview(encoded)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("upgrade journal write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(temporary, path)
        published = True
        _fsync_directory(path.parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if not published:
            temporary.unlink(missing_ok=True)


def _read_journal(path: Path) -> dict[str, object]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("agent-home upgrade journal is not a regular file")
        chunks: list[bytes] = []
        size = 0
        while size <= _MAX_JOURNAL_BYTES:
            chunk = os.read(descriptor, min(8_192, _MAX_JOURNAL_BYTES + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
    if len(encoded) > _MAX_JOURNAL_BYTES:
        raise ValueError("agent-home upgrade journal exceeds its bound")
    value = json.loads(encoded.decode("utf-8"))
    if not isinstance(value, dict):
        raise TypeError("agent-home upgrade journal is invalid")
    return value


def _copy_regular(source: Path, destination: Path) -> None:
    _require_regular(source)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with source.open("rb") as read, destination.open("xb") as write:
        shutil.copyfileobj(read, write, length=1024 * 1024)
        write.flush()
        os.fsync(write.fileno())
    os.chmod(destination, 0o600)
    _fsync_directory(destination.parent)


def _copy_database(source: Path, destination: Path) -> None:
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    source_connection = _connect_read_only(source)
    destination_connection = sqlite3.connect(destination)
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()
        source_connection.close()
    os.chmod(destination, 0o600)
    with _connect_read_only(destination) as copied:
        require_healthy(copied)
    _fsync_directory(destination.parent)


def _stage_file(source: Path, destination: Path, *, database: bool) -> None:
    if database:
        _copy_database(source, destination)
    else:
        _copy_regular(source, destination)


def _preflight(home: Path, affected_paths: tuple[str, ...]) -> None:
    required = _DISK_HEADROOM_BYTES
    for relative in affected_paths:
        path = _contained_home_path(home, relative)
        if not path.exists():
            continue
        _require_regular(path)
        required += path.stat().st_size * 2
    if shutil.disk_usage(home).free < required:
        raise OSError("insufficient free space for a recoverable agent-home upgrade")


def _planned_migrations(source_revision: int) -> tuple[HomeMigration, ...]:
    return tuple(
        migration
        for migration in HOME_MIGRATIONS
        if migration.revision > source_revision
    )


def _journal_path(value: object, allowed: frozenset[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError("agent-home upgrade journal path is invalid")
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("agent-home upgrade journal path is unsafe")
    return value


def _journal_digest(value: object, *, optional: bool) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("agent-home upgrade journal digest is invalid")
    return value


def _validate_journal(journal: dict[str, object]) -> tuple[str, ...]:
    if set(journal) != {
        "files",
        "kind",
        "migrations",
        "operation_id",
        "phase",
        "source_kind",
        "source_revision",
        "target_revision",
    }:
        raise ValueError("agent-home upgrade journal fields are invalid")
    if journal.get("kind") != "daita_agent_home_upgrade":
        raise ValueError("agent-home upgrade journal kind is invalid")
    if journal.get("source_kind") not in {
        "production",
        "preproduction_current",
        "preproduction_inbox",
        "preproduction_routines",
    }:
        raise ValueError("agent-home upgrade source kind is invalid")
    operation_id = journal.get("operation_id")
    if (
        not isinstance(operation_id, str)
        or len(operation_id) != 32
        or any(character not in "0123456789abcdef" for character in operation_id)
    ):
        raise ValueError("agent-home upgrade operation identity is invalid")
    source_revision = journal.get("source_revision")
    target_revision = journal.get("target_revision")
    if (
        not isinstance(source_revision, int)
        or isinstance(source_revision, bool)
        or source_revision < 0
        or not isinstance(target_revision, int)
        or isinstance(target_revision, bool)
    ):
        raise ValueError("agent-home upgrade revisions are invalid")
    if target_revision > CURRENT_HOME_REVISION:
        raise HomeMigrationJournalNewerError(
            "unfinished upgrade belongs to a newer release", target_revision
        )
    if target_revision != CURRENT_HOME_REVISION or source_revision >= target_revision:
        raise ValueError("agent-home upgrade revision range is invalid")
    planned = _planned_migrations(source_revision)
    raw_migrations = journal.get("migrations")
    if not isinstance(raw_migrations, list) or len(raw_migrations) != len(planned):
        raise ValueError("agent-home upgrade migration plan is invalid")
    phase = journal.get("phase")
    if phase not in {"staging", "prepared", "committing", "rolling_back", "committed"}:
        raise ValueError("agent-home upgrade journal phase is invalid")
    files = journal.get("files")
    if not isinstance(files, list):
        raise TypeError("agent-home upgrade file journal is invalid")
    affected = tuple(
        dict.fromkeys(
            relative for migration in planned for relative in migration.affected_paths
        )
    )
    # Nothing in the active home can have changed while the journal is still in
    # staging. A later release may therefore discard this private work area even
    # when the interrupted migration implementation no longer has the same hash.
    if phase == "staging":
        if files:
            raise ValueError("staging journal cannot claim prepared files")
        return affected
    for raw, migration in zip(raw_migrations, planned, strict=True):
        if not isinstance(raw, dict) or set(raw) != {
            "checksum",
            "migration_id",
            "revision",
        }:
            raise ValueError("agent-home upgrade migration entry is invalid")
        if raw != {
            "checksum": migration.checksum,
            "migration_id": migration.migration_id,
            "revision": migration.revision,
        }:
            raise ValueError("agent-home upgrade migration plan was changed")
    allowed = frozenset(affected)
    if len(files) != len(affected):
        raise ValueError("agent-home upgrade file inventory is incomplete")
    observed: list[str] = []
    for raw in files:
        if not isinstance(raw, dict) or set(raw) != {
            "backup_sha256",
            "path",
            "source_exists",
            "source_sha256",
            "target_exists",
            "target_sha256",
        }:
            raise ValueError("agent-home upgrade file entry is invalid")
        relative = _journal_path(raw["path"], allowed)
        if not isinstance(raw["source_exists"], bool) or not isinstance(
            raw["target_exists"], bool
        ):
            raise TypeError("agent-home upgrade file presence is invalid")
        source_digest = _journal_digest(
            raw["source_sha256"], optional=not raw["source_exists"]
        )
        backup_digest = _journal_digest(
            raw["backup_sha256"], optional=not raw["source_exists"]
        )
        target_digest = _journal_digest(
            raw["target_sha256"], optional=not raw["target_exists"]
        )
        if raw["source_exists"] != (source_digest is not None):
            raise ValueError("agent-home upgrade source digest is inconsistent")
        if raw["source_exists"] != (backup_digest is not None):
            raise ValueError("agent-home upgrade backup digest is inconsistent")
        if raw["target_exists"] != (target_digest is not None):
            raise ValueError("agent-home upgrade target digest is inconsistent")
        observed.append(relative)
    if tuple(observed) != affected:
        raise ValueError("agent-home upgrade file order is invalid")
    return affected


def _stage_upgrade(
    home: Path,
    status: AgentHomeStatus,
    *,
    validate_home: HomeValidator,
    phase_hook: PhaseHook | None,
) -> dict[str, object]:
    source_revision = status.found_revision or 0
    migrations = _planned_migrations(source_revision)
    if not migrations:
        raise RuntimeError("agent-home upgrade plan is empty")
    affected_paths = tuple(
        dict.fromkeys(
            relative
            for migration in migrations
            for relative in migration.affected_paths
        )
    )
    _preflight(home, affected_paths)
    upgrade = home / _UPGRADE_DIRECTORY
    upgrade.mkdir(mode=0o700)
    stage = upgrade / "stage"
    backup = upgrade / "backup"
    stage.mkdir(mode=0o700)
    backup.mkdir(mode=0o700)
    journal: dict[str, object] = {
        "kind": "daita_agent_home_upgrade",
        "operation_id": uuid4().hex,
        "phase": "staging",
        "source_kind": status.source_kind,
        "source_revision": source_revision,
        "target_revision": CURRENT_HOME_REVISION,
        "migrations": [
            {
                "revision": migration.revision,
                "migration_id": migration.migration_id,
                "checksum": migration.checksum,
            }
            for migration in migrations
        ],
        "files": [],
    }
    _write_json(upgrade / _JOURNAL_NAME, journal)
    if phase_hook is not None:
        phase_hook("staging")
    files: list[dict[str, object]] = []
    for relative in affected_paths:
        source = _contained_home_path(home, relative)
        source_exists = source.exists()
        if source_exists:
            _stage_file(source, backup / relative, database=relative == "state.db")
            _stage_file(source, stage / relative, database=relative == "state.db")
        elif relative == "state.db":
            raise ValueError("agent-home state database is missing")
        files.append(
            {
                "path": relative,
                "source_exists": source_exists,
                "source_sha256": _sha256(source) if source_exists else None,
                "backup_sha256": (
                    _sha256(backup / relative) if source_exists else None
                ),
            }
        )
    source_shape = None if status.source_kind == "production" else status.source_kind
    for migration in migrations:
        migration.apply(stage, source_shape)
        with sqlite3.connect(stage / "state.db") as connection:
            insert_migration_row(connection, migration)
            require_schema(connection, migration.target_schema)
            require_healthy(connection)
            connection.commit()
            checkpoint = connection.execute(
                "PRAGMA wal_checkpoint(TRUNCATE)"
            ).fetchone()
            if (
                checkpoint is None
                or int(checkpoint[0]) != 0
                or int(checkpoint[1]) != int(checkpoint[2])
            ):
                raise ValueError("staged migration WAL checkpoint did not complete")
        source_shape = None
    validate_home(home, stage, frozenset(affected_paths))
    for entry in files:
        relative = str(entry["path"])
        target = stage / relative
        entry["target_exists"] = target.exists()
        entry["target_sha256"] = _sha256(target) if target.exists() else None
    journal["files"] = files
    journal["phase"] = "prepared"
    _write_json(upgrade / _JOURNAL_NAME, journal)
    if phase_hook is not None:
        phase_hook("prepared")
    return journal


def _atomic_publish(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.upgrade")
    try:
        _copy_regular(source, temporary)
        os.replace(temporary, destination)
        if destination.name == "state.db":
            destination.with_name("state.db-wal").unlink(missing_ok=True)
            destination.with_name("state.db-shm").unlink(missing_ok=True)
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _file_hash_or_none(path: Path) -> str | None:
    if path.is_symlink():
        raise ValueError(f"agent-home file is a symlink: {path.name}")
    return _sha256(path) if path.exists() else None


def _restore_source(home: Path, journal: dict[str, object]) -> None:
    upgrade = home / _UPGRADE_DIRECTORY
    backup = upgrade / "backup"
    affected = frozenset(_validate_journal(journal))
    files = journal.get("files")
    if not isinstance(files, list):
        raise TypeError("agent-home upgrade file journal is invalid")
    journal["phase"] = "rolling_back"
    _write_json(upgrade / _JOURNAL_NAME, journal)
    for raw in reversed(files):
        if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
            raise TypeError("agent-home upgrade file entry is invalid")
        relative = _journal_path(raw["path"], affected)
        destination = _contained_home_path(home, relative)
        if raw.get("source_exists") is True:
            backup_path = backup / relative
            if _sha256(backup_path) != raw.get("backup_sha256"):
                raise ValueError(f"agent-home rollback backup is invalid: {relative}")
            _atomic_publish(backup_path, destination)
            if _sha256(destination) != raw.get("backup_sha256"):
                raise RuntimeError(f"agent-home rollback did not persist: {relative}")
        else:
            destination.unlink(missing_ok=True)
            _fsync_directory(destination.parent)


def _retain_rollback(home: Path, journal: dict[str, object]) -> Path:
    upgrade = home / _UPGRADE_DIRECTORY
    root = home / _ROLLBACK_DIRECTORY
    root.mkdir(mode=0o700, exist_ok=True)
    root_facts = root.lstat()
    if not stat.S_ISDIR(root_facts.st_mode) or stat.S_ISLNK(root_facts.st_mode):
        raise ValueError("agent-home rollback root is invalid")
    operation_id = journal.get("operation_id")
    source_revision = journal.get("source_revision")
    if not isinstance(operation_id, str) or not isinstance(source_revision, int):
        raise TypeError("agent-home upgrade journal identity is invalid")
    retained = root / f"revision-{source_revision}-{operation_id}"
    backup = upgrade / "backup"
    if retained.is_symlink():
        raise ValueError("retained rollback path is invalid")
    if retained.exists():
        if retained.is_symlink() or not retained.is_dir():
            raise ValueError("retained rollback path is invalid")
        if backup.exists():
            raise ValueError("agent-home rollback exists in two locations")
    else:
        if backup.is_symlink() or not backup.is_dir():
            raise ValueError("agent-home upgrade backup is unavailable")
        os.replace(backup, retained)
    _write_json(retained / "rollback.json", journal)
    _fsync_directory(root)
    rollbacks = list(root.iterdir())
    for item in rollbacks:
        facts = item.lstat()
        if (
            not stat.S_ISDIR(facts.st_mode)
            or stat.S_ISLNK(facts.st_mode)
            or re.fullmatch(r"revision-[0-9]+-[0-9a-f]{32}", item.name) is None
        ):
            raise ValueError("agent-home rollback inventory is invalid")
    rollbacks.sort(key=lambda item: item.stat().st_mtime_ns)
    for obsolete in rollbacks[:-1]:
        if obsolete.is_dir() and not obsolete.is_symlink():
            shutil.rmtree(obsolete)
    return retained


def _finish_commit(
    home: Path,
    journal: dict[str, object],
    *,
    validate_home: HomeValidator,
    phase_hook: PhaseHook | None,
) -> Path:
    upgrade = home / _UPGRADE_DIRECTORY
    stage = upgrade / "stage"
    affected = frozenset(_validate_journal(journal))
    files = journal.get("files")
    if not isinstance(files, list):
        raise TypeError("agent-home upgrade file journal is invalid")
    journal["phase"] = "committing"
    _write_json(upgrade / _JOURNAL_NAME, journal)
    if phase_hook is not None:
        phase_hook("committing")
    ordered = sorted(
        files,
        key=lambda item: (
            1 if isinstance(item, dict) and item.get("path") == "state.db" else 0
        ),
    )
    try:
        for raw in ordered:
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                raise TypeError("agent-home upgrade file entry is invalid")
            relative = _journal_path(raw["path"], affected)
            destination = _contained_home_path(home, relative)
            actual = _file_hash_or_none(destination)
            if actual == raw.get("target_sha256"):
                continue
            if actual != raw.get("source_sha256"):
                raise RuntimeError(
                    f"agent-home file changed during upgrade: {relative}"
                )
            if raw.get("target_exists") is True:
                staged_path = stage / relative
                if _sha256(staged_path) != raw.get("target_sha256"):
                    raise ValueError(f"staged agent-home file is invalid: {relative}")
                _atomic_publish(staged_path, destination)
                if _sha256(destination) != raw.get("target_sha256"):
                    raise RuntimeError(
                        f"agent-home file did not persist after commit: {relative}"
                    )
            else:
                destination.unlink(missing_ok=True)
                _fsync_directory(destination.parent)
        validate_home(home, home, affected)
    except BaseException:
        _restore_source(home, journal)
        raise
    journal["phase"] = "committed"
    _write_json(upgrade / _JOURNAL_NAME, journal)
    if phase_hook is not None:
        phase_hook("committed")
    retained = _retain_rollback(home, journal)
    shutil.rmtree(stage)
    (upgrade / _JOURNAL_NAME).unlink()
    upgrade.rmdir()
    _fsync_directory(home)
    return retained


def _recover_upgrade(
    home: Path,
    *,
    validate_home: HomeValidator,
    phase_hook: PhaseHook | None,
) -> tuple[bool, Path | None]:
    upgrade = home / _UPGRADE_DIRECTORY
    if not upgrade.exists() and not upgrade.is_symlink():
        return False, None
    if upgrade.is_symlink() or not upgrade.is_dir():
        raise ValueError("agent-home upgrade recovery path is invalid")
    journal_path = upgrade / _JOURNAL_NAME
    if not journal_path.exists() and not journal_path.is_symlink():
        with os.scandir(upgrade) as iterator:
            entries = tuple(iterator)
        if entries:
            raise ValueError("agent-home upgrade journal is missing")
        upgrade.rmdir()
        _fsync_directory(home)
        return True, None
    journal = _read_journal(journal_path)
    _validate_journal(journal)
    phase = journal.get("phase")
    if phase == "staging":
        shutil.rmtree(upgrade)
        _fsync_directory(home)
        return True, None
    if phase in {"prepared", "committing"}:
        return True, _finish_commit(
            home,
            journal,
            validate_home=validate_home,
            phase_hook=phase_hook,
        )
    if phase == "rolling_back":
        _restore_source(home, journal)
        shutil.rmtree(upgrade)
        _fsync_directory(home)
        return True, None
    if phase == "committed":
        validate_home(home, home, frozenset(_validate_journal(journal)))
        retained = _retain_rollback(home, journal)
        stage = upgrade / "stage"
        if stage.exists():
            shutil.rmtree(stage)
        (upgrade / _JOURNAL_NAME).unlink()
        upgrade.rmdir()
        _fsync_directory(home)
        return True, retained
    raise ValueError("agent-home upgrade journal phase is invalid")


def upgrade_agent_home(
    home: Path,
    *,
    validate_home: HomeValidator,
    phase_hook: PhaseHook | None = None,
) -> AgentHomeUpgradeResult:
    """Recover or upgrade one locked home and return its exact transition."""

    home = _resolve_home(home)
    recovered = False
    try:
        recovered, recovered_rollback = _recover_upgrade(
            home,
            validate_home=validate_home,
            phase_hook=phase_hook,
        )
        status = inspect_agent_home(home)
        if not status.upgrade_required:
            try:
                validate_home(home, home, frozenset())
            except StateCompatibilityError:
                raise
            except BaseException as error:
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
                raise _error(
                    StateCompatibilityCode.DAMAGED,
                    home,
                    "This agent home or model configuration does not match its "
                    "declared revision. No local data was changed.",
                    found=status.found_revision,
                ) from error
            return AgentHomeUpgradeResult(
                source_revision=status.found_revision or CURRENT_HOME_REVISION,
                target_revision=CURRENT_HOME_REVISION,
                upgraded=False,
                recovered=recovered,
                rollback_path=recovered_rollback,
            )
        source_revision = status.found_revision or 0
        journal = _stage_upgrade(
            home,
            status,
            validate_home=validate_home,
            phase_hook=phase_hook,
        )
        rollback = _finish_commit(
            home,
            journal,
            validate_home=validate_home,
            phase_hook=phase_hook,
        )
        return AgentHomeUpgradeResult(
            source_revision=source_revision,
            target_revision=CURRENT_HOME_REVISION,
            upgraded=True,
            recovered=recovered,
            rollback_path=rollback,
        )
    except HomeMigrationJournalNewerError as error:
        raise _error(
            StateCompatibilityCode.NEWER_REVISION,
            home,
            "This unfinished agent-home upgrade belongs to a newer Daita "
            "release. No local data was changed.",
            found=error.found_revision,
        ) from None
    except StateCompatibilityError:
        raise
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise _error(
            StateCompatibilityCode.UPGRADE_FAILED,
            home,
            "Daita could not upgrade this agent home safely. The previous home "
            "was retained or restored; no reset is required.",
            found=None,
        ) from error


__all__ = [
    "AgentHomeStatus",
    "AgentHomeUpgradeResult",
    "inspect_agent_home",
    "upgrade_agent_home",
]
