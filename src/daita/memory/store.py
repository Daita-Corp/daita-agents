"""Safe fixed-file storage for bounded advisory agent memory."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

MEMORY_MAX_CHARACTERS = 2_200
MEMORY_MAX_UTF8_BYTES = 8_800
USER_MAX_CHARACTERS = 1_375
USER_MAX_UTF8_BYTES = 5_500

_MEMORY_NAME = "MEMORY.md"
_USER_NAME = "USER.md"
_T = TypeVar("_T")


class MemoryStoreError(Exception):
    """Base failure for an owned advisory document."""


class MemoryValidationError(MemoryStoreError, ValueError):
    """The complete document violates its encoding or size contract."""


class MemoryPathError(MemoryStoreError):
    """An owned path no longer has its admitted safe shape."""


class MemoryStore:
    """Own exactly MEMORY.md and USER.md beneath one admitted agent home."""

    def __init__(self, agent_home: Path, mutation_lock: asyncio.Lock) -> None:
        if not isinstance(agent_home, Path):
            raise TypeError("agent_home must be a Path")
        if not agent_home.is_absolute() or ".." in agent_home.parts:
            raise MemoryPathError("agent home must be an absolute contained path")
        if not isinstance(mutation_lock, asyncio.Lock):
            raise TypeError("mutation_lock must be an asyncio.Lock")
        home = Path(os.path.abspath(os.fspath(agent_home)))
        try:
            home_state = os.lstat(home)
        except OSError as error:
            raise MemoryPathError("agent home is unavailable") from error
        if not stat.S_ISDIR(home_state.st_mode) or stat.S_ISLNK(home_state.st_mode):
            raise MemoryPathError("agent home must be a non-symlink directory")
        if home.resolve(strict=True) != home:
            raise MemoryPathError("agent home cannot contain a symlink or path alias")
        self._home = home
        self._home_identity = (home_state.st_dev, home_state.st_ino)
        self._mutation_lock = mutation_lock
        self._closed = False

    async def read_memory(self) -> str:
        return await self._read(
            _MEMORY_NAME,
            MEMORY_MAX_CHARACTERS,
            MEMORY_MAX_UTF8_BYTES,
        )

    async def set_memory(self, text: str) -> None:
        await self._write(
            _MEMORY_NAME,
            text,
            MEMORY_MAX_CHARACTERS,
            MEMORY_MAX_UTF8_BYTES,
        )

    async def read_user_profile(self) -> str:
        return await self._read(
            _USER_NAME,
            USER_MAX_CHARACTERS,
            USER_MAX_UTF8_BYTES,
        )

    async def set_user_profile(self, text: str) -> None:
        await self._write(
            _USER_NAME,
            text,
            USER_MAX_CHARACTERS,
            USER_MAX_UTF8_BYTES,
        )

    async def preflight_replacement(
        self,
        target: str,
        content: str,
    ) -> tuple[bool, str, str]:
        """Validate a proposed complete replacement and fingerprint current state."""

        name, max_characters, max_bytes = _target_contract(target)
        _validate_text(content, max_characters, max_bytes)
        self._require_open()
        return await asyncio.to_thread(
            self._preflight_replacement_sync,
            name,
            max_characters,
            max_bytes,
        )

    async def replace_from_tool(self, target: str, content: str) -> None:
        """Replace after runtime authorization while the shared lock is held."""

        if not self._mutation_lock.locked():
            raise MemoryStoreError("tool replacement requires the mutation lock")
        name, max_characters, max_bytes = _target_contract(target)
        data = _validate_text(content, max_characters, max_bytes)
        self._require_open()
        await asyncio.to_thread(
            self._write_sync,
            name,
            content,
            data,
            max_characters,
            max_bytes,
        )

    async def close(self) -> None:
        async with self._mutation_lock:
            self._closed = True

    async def _read(self, name: str, max_characters: int, max_bytes: int) -> str:
        async with self._mutation_lock:
            self._require_open()
            value, cancelled = await _await_sync_completion(
                lambda: self._read_sync(name, max_characters, max_bytes)
            )
        if cancelled:
            raise asyncio.CancelledError
        return value

    async def _write(
        self,
        name: str,
        text: str,
        max_characters: int,
        max_bytes: int,
    ) -> None:
        data = _validate_text(text, max_characters, max_bytes)
        async with self._mutation_lock:
            self._require_open()
            _, cancelled = await _await_sync_completion(
                lambda: self._write_sync(
                    name,
                    text,
                    data,
                    max_characters,
                    max_bytes,
                )
            )
        if cancelled:
            raise asyncio.CancelledError

    def _require_open(self) -> None:
        if self._closed:
            raise MemoryStoreError("memory store is closed")

    def _read_sync(self, name: str, max_characters: int, max_bytes: int) -> str:
        directory = self._open_home()
        try:
            text, _ = _read_owned(directory, name, max_characters, max_bytes)
            return text
        finally:
            os.close(directory)

    def _write_sync(
        self,
        name: str,
        text: str,
        data: bytes,
        max_characters: int,
        max_bytes: int,
    ) -> None:
        directory = self._open_home()
        temporary = f".{name}.{uuid4().hex}.tmp"
        temporary_created = False
        try:
            prior_text, prior_state = _read_owned(
                directory,
                name,
                max_characters,
                max_bytes,
            )
            del prior_text
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
            temporary_created = True
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise MemoryPathError("temporary memory file is not private")
                with os.fdopen(descriptor, "wb") as file:
                    file.write(data)
                    file.flush()
                    os.fsync(file.fileno())
            except BaseException:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
                raise
            _require_unchanged_target(directory, name, prior_state)
            os.replace(
                temporary,
                name,
                src_dir_fd=directory,
                dst_dir_fd=directory,
            )
            temporary_created = False
            written, _ = _read_owned(directory, name, max_characters, max_bytes)
            if written != text:
                raise MemoryStoreError("memory replacement did not round-trip exactly")
        except (MemoryStoreError, MemoryValidationError):
            raise
        except OSError as error:
            raise MemoryPathError(
                f"cannot replace owned memory document {name}"
            ) from error
        finally:
            if temporary_created:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
            os.close(directory)

    def _preflight_replacement_sync(
        self,
        name: str,
        max_characters: int,
        max_bytes: int,
    ) -> tuple[bool, str, str]:
        directory = self._open_home()
        try:
            current, state = _read_owned(
                directory,
                name,
                max_characters,
                max_bytes,
            )
            state_fingerprint = (
                "absent"
                if state is None
                else sha256(
                    repr(
                        (
                            state.st_dev,
                            state.st_ino,
                            state.st_mode,
                            state.st_nlink,
                            state.st_size,
                            state.st_mtime_ns,
                            state.st_ctime_ns,
                        )
                    ).encode("ascii")
                ).hexdigest()
            )
            return (
                state is not None,
                sha256(current.encode("utf-8")).hexdigest(),
                state_fingerprint,
            )
        finally:
            os.close(directory)

    def _open_home(self) -> int:
        try:
            lexical = os.lstat(self._home)
            if (
                not stat.S_ISDIR(lexical.st_mode)
                or stat.S_ISLNK(lexical.st_mode)
                or (lexical.st_dev, lexical.st_ino) != self._home_identity
                or self._home.resolve(strict=True) != self._home
            ):
                raise MemoryPathError("agent home path identity changed")
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self._home, flags)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != self._home_identity
            ):
                os.close(descriptor)
                raise MemoryPathError("agent home directory identity changed")
            return descriptor
        except MemoryPathError:
            raise
        except OSError as error:
            raise MemoryPathError("agent home path is invalid") from error


def _validate_text(text: str, max_characters: int, max_bytes: int) -> bytes:
    if not isinstance(text, str):
        raise TypeError("memory content must be text")
    data = text.encode("utf-8")
    if len(data) > max_bytes:
        raise MemoryValidationError(
            f"memory content exceeds the {max_bytes} UTF-8 byte limit"
        )
    if len(text) > max_characters:
        raise MemoryValidationError(
            f"memory content exceeds the {max_characters} character limit"
        )
    return data


def _target_contract(target: str) -> tuple[str, int, int]:
    if target == "memory":
        return _MEMORY_NAME, MEMORY_MAX_CHARACTERS, MEMORY_MAX_UTF8_BYTES
    if target == "user":
        return _USER_NAME, USER_MAX_CHARACTERS, USER_MAX_UTF8_BYTES
    raise MemoryValidationError("memory target must be memory or user")


def _read_owned(
    directory: int,
    name: str,
    max_characters: int,
    max_bytes: int,
) -> tuple[str, os.stat_result | None]:
    state = _target_state(directory, name)
    if state is None:
        return "", None
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory)
    except OSError as error:
        raise MemoryPathError(f"cannot open owned memory document {name}") from error
    try:
        opened = os.fstat(descriptor)
        _require_same_state(state, opened, name)
        with os.fdopen(descriptor, "rb") as file:
            data = file.read(max_bytes + 1)
            final = os.fstat(file.fileno())
        descriptor = -1
        _require_same_state(state, final, name)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(data) > max_bytes:
        raise MemoryValidationError(f"{name} exceeds the {max_bytes} UTF-8 byte limit")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise MemoryValidationError(f"{name} is not strict UTF-8") from error
    _validate_text(text, max_characters, max_bytes)
    return text, state


def _target_state(directory: int, name: str) -> os.stat_result | None:
    try:
        state = os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise MemoryPathError(f"cannot inspect owned memory document {name}") from error
    if not stat.S_ISREG(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise MemoryPathError(f"{name} must be a regular non-symlink file")
    if state.st_nlink != 1:
        raise MemoryPathError(f"{name} must have exactly one hard link")
    return state


def _require_unchanged_target(
    directory: int,
    name: str,
    expected: os.stat_result | None,
) -> None:
    current = _target_state(directory, name)
    if expected is None:
        if current is not None:
            raise MemoryPathError(f"{name} path changed before replacement")
        return
    if current is None:
        raise MemoryPathError(f"{name} disappeared before replacement")
    _require_same_state(expected, current, name)


def _require_same_state(
    expected: os.stat_result,
    current: os.stat_result,
    name: str,
) -> None:
    before = (
        expected.st_dev,
        expected.st_ino,
        expected.st_mode,
        expected.st_nlink,
        expected.st_size,
        expected.st_mtime_ns,
        expected.st_ctime_ns,
    )
    after = (
        current.st_dev,
        current.st_ino,
        current.st_mode,
        current.st_nlink,
        current.st_size,
        current.st_mtime_ns,
        current.st_ctime_ns,
    )
    if before != after:
        raise MemoryPathError(f"{name} path or file state changed during access")


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


__all__ = [
    "MEMORY_MAX_CHARACTERS",
    "MEMORY_MAX_UTF8_BYTES",
    "USER_MAX_CHARACTERS",
    "USER_MAX_UTF8_BYTES",
    "MemoryPathError",
    "MemoryStore",
    "MemoryStoreError",
    "MemoryValidationError",
]
