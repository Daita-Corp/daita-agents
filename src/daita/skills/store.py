"""Safe fixed-layout storage for bounded procedural skill documents."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
from typing import TypeVar
from uuid import uuid4

SKILL_MAX_COUNT = 32
SKILL_DESCRIPTION_MAX_CHARACTERS = 240
SKILL_INSTRUCTIONS_MAX_CHARACTERS = 12_000
SKILL_INDEX_MAX_CHARACTERS = 4_000
SKILL_RENDERED_MAX_UTF8_BYTES = 50_000
SKILL_INDEX_MAX_UTF8_BYTES = 16_000

_SKILLS_DIRECTORY = "skills"
_SKILL_DOCUMENT = "SKILL.md"
_SKILL_NAME = re.compile(r"[a-z][a-z0-9-]{0,63}\Z", re.ASCII)
_T = TypeVar("_T")


class SkillStoreError(Exception):
    """Base failure for bounded skill storage."""


class SkillValidationError(SkillStoreError, ValueError):
    """A skill document or complete skill index violates its contract."""


class SkillPathError(SkillStoreError):
    """An owned skill path no longer has its admitted safe shape."""


class SkillNotFoundError(SkillStoreError, LookupError):
    """A validated skill name is not present."""


@dataclass(frozen=True, slots=True)
class SkillSummary:
    name: str
    description: str

    def __post_init__(self) -> None:
        validate_skill_name(self.name)
        _validate_description(self.description)


@dataclass(frozen=True, slots=True)
class Skill:
    name: str
    description: str
    instructions: str

    def __post_init__(self) -> None:
        validate_skill_name(self.name)
        _validate_description(self.description)
        _validate_instructions(self.instructions)
        _render_skill(self)

    @property
    def summary(self) -> SkillSummary:
        return SkillSummary(self.name, self.description)


class SkillStore:
    """Own one deterministic tree of bounded ``SKILL.md`` documents."""

    def __init__(self, agent_home: Path, mutation_lock: asyncio.Lock) -> None:
        if not isinstance(agent_home, Path):
            raise TypeError("agent_home must be a Path")
        if not agent_home.is_absolute() or ".." in agent_home.parts:
            raise SkillPathError("agent home must be an absolute contained path")
        if not isinstance(mutation_lock, asyncio.Lock):
            raise TypeError("mutation_lock must be an asyncio.Lock")
        home = Path(os.path.abspath(os.fspath(agent_home)))
        try:
            home_state = os.lstat(home)
        except OSError as error:
            raise SkillPathError("agent home is unavailable") from error
        if not stat.S_ISDIR(home_state.st_mode) or stat.S_ISLNK(home_state.st_mode):
            raise SkillPathError("agent home must be a non-symlink directory")
        if home.resolve(strict=True) != home:
            raise SkillPathError("agent home cannot contain a symlink or path alias")
        self._home = home
        self._home_identity = (home_state.st_dev, home_state.st_ino)
        self._mutation_lock = mutation_lock
        self._closed = False

    async def list_skills(self) -> tuple[SkillSummary, ...]:
        skills = await self._run_locked(self._list_sync)
        return tuple(item.summary for item in skills)

    async def read_skill(self, name: str) -> Skill | None:
        validate_skill_name(name)
        return await self._run_locked(lambda: self._read_sync(name))

    async def read_skill_with_digest(self, name: str) -> tuple[Skill | None, str]:
        """Read one skill and its preflight rendered-document digest atomically."""

        validate_skill_name(name)
        inspected = await self._run_locked(
            lambda: self._inspect_sync(name, None, False)
        )
        return inspected[0], inspected[2]

    async def skill_index(self) -> str:
        skills = await self._run_locked(self._list_sync)
        return render_skill_index(item.summary for item in skills)

    async def save_skill(
        self,
        name: str,
        description: str,
        instructions: str,
    ) -> bool:
        skill = Skill(name, description, instructions)
        rendered = _render_skill(skill)
        return await self._run_locked(lambda: self._save_sync(skill, rendered))

    async def delete_skill(self, name: str) -> bool:
        validate_skill_name(name)
        return await self._run_locked(lambda: self._delete_sync(name))

    async def preflight_save(
        self,
        name: str,
        description: str,
        instructions: str,
    ) -> tuple[bool, str, str, str]:
        """Validate one save and fingerprint its document and complete index."""

        skill = Skill(name, description, instructions)
        self._require_open()
        _selected, exists, digest, state_digest, index_digest = await asyncio.to_thread(
            self._inspect_sync,
            name,
            skill,
            False,
        )
        return exists, digest, state_digest, index_digest

    async def preflight_delete(self, name: str) -> tuple[bool, str, str, str]:
        """Validate one deletion and fingerprint its document and complete index."""

        validate_skill_name(name)
        self._require_open()
        _selected, exists, digest, state_digest, index_digest = await asyncio.to_thread(
            self._inspect_sync,
            name,
            None,
            True,
        )
        return exists, digest, state_digest, index_digest

    async def save_from_tool(
        self,
        name: str,
        description: str,
        instructions: str,
    ) -> bool:
        """Save after runtime authorization while the shared lock is held."""

        if not self._mutation_lock.locked():
            raise SkillStoreError("tool save requires the mutation lock")
        skill = Skill(name, description, instructions)
        rendered = _render_skill(skill)
        self._require_open()
        return await asyncio.to_thread(self._save_sync, skill, rendered)

    async def delete_from_tool(self, name: str) -> bool:
        """Delete after runtime authorization while the shared lock is held."""

        if not self._mutation_lock.locked():
            raise SkillStoreError("tool deletion requires the mutation lock")
        validate_skill_name(name)
        self._require_open()
        return await asyncio.to_thread(self._delete_sync, name)

    async def close(self) -> None:
        async with self._mutation_lock:
            self._closed = True

    async def _run_locked(self, callback: Callable[[], _T]) -> _T:
        async with self._mutation_lock:
            self._require_open()
            value, cancelled = await _await_sync_completion(callback)
        if cancelled:
            raise asyncio.CancelledError
        return value

    def _require_open(self) -> None:
        if self._closed:
            raise SkillStoreError("skill store is closed")

    def _list_sync(self) -> tuple[Skill, ...]:
        home = self._open_home()
        try:
            root, root_state = _open_directory(home, _SKILLS_DIRECTORY, required=False)
            if root is None:
                return ()
            try:
                skills = _list_from_root(root)
                _require_directory_identity(home, _SKILLS_DIRECTORY, root_state)
                return skills
            finally:
                os.close(root)
        finally:
            os.close(home)

    def _read_sync(self, name: str) -> Skill | None:
        home = self._open_home()
        try:
            root, root_state = _open_directory(home, _SKILLS_DIRECTORY, required=False)
            if root is None:
                return None
            try:
                skill = next(
                    (skill for skill in _list_from_root(root) if skill.name == name),
                    None,
                )
                _require_directory_identity(home, _SKILLS_DIRECTORY, root_state)
                return skill
            finally:
                os.close(root)
        finally:
            os.close(home)

    def _inspect_sync(
        self,
        name: str,
        candidate: Skill | None,
        require_present: bool,
    ) -> tuple[Skill | None, bool, str, str, str]:
        home = self._open_home()
        try:
            root, root_state = _open_directory(home, _SKILLS_DIRECTORY, required=False)
            if root is None:
                if require_present:
                    raise SkillNotFoundError(name)
                current: tuple[Skill, ...] = ()
                current_index = render_skill_index(())
                selected = None
                selected_bytes = b""
                selected_state = "absent"
            else:
                try:
                    current = _list_from_root(root)
                    current_index = render_skill_index(item.summary for item in current)
                    by_name = {item.name: item for item in current}
                    selected = by_name.get(name)
                    if selected is None:
                        if require_present:
                            raise SkillNotFoundError(name)
                        selected_bytes = b""
                        selected_state = "absent"
                    else:
                        directory, directory_state = _open_directory(
                            root, name, required=True
                        )
                        assert directory is not None and directory_state is not None
                        try:
                            checked, selected_bytes, document_state = (
                                _read_skill_document(directory, name)
                            )
                            if checked != selected:
                                raise SkillPathError(
                                    "skill content changed during preflight"
                                )
                            selected_state = _state_fingerprint(
                                directory_state,
                                document_state,
                            )
                        finally:
                            os.close(directory)
                            _require_directory_identity(root, name, directory_state)
                    _require_directory_identity(home, _SKILLS_DIRECTORY, root_state)
                finally:
                    os.close(root)

            if candidate is not None:
                by_name = {item.name: item for item in current}
                proposed = dict(by_name)
                proposed[candidate.name] = candidate
                if candidate.name not in by_name and len(proposed) > SKILL_MAX_COUNT:
                    raise SkillValidationError(
                        f"skill count exceeds the {SKILL_MAX_COUNT} skill limit"
                    )
                render_skill_index(item.summary for item in proposed.values())

            return (
                selected,
                bool(selected_bytes),
                _rendered_document_sha256(selected_bytes),
                selected_state,
                sha256(current_index.encode("utf-8")).hexdigest(),
            )
        finally:
            os.close(home)

    def _save_sync(self, skill: Skill, rendered: bytes) -> bool:
        home = self._open_home()
        root: int | None = None
        root_created = False
        directory: int | None = None
        directory_created = False
        temporary: str | None = None
        replaced = False
        try:
            root, root_state = _open_directory(home, _SKILLS_DIRECTORY, required=False)
            current = () if root is None else _list_from_root(root)
            by_name = {item.name: item for item in current}
            candidate = dict(by_name)
            candidate[skill.name] = skill
            if skill.name not in by_name and len(candidate) > SKILL_MAX_COUNT:
                raise SkillValidationError(
                    f"skill count exceeds the {SKILL_MAX_COUNT} skill limit"
                )
            render_skill_index(item.summary for item in candidate.values())

            if root is None:
                try:
                    os.mkdir(_SKILLS_DIRECTORY, mode=0o700, dir_fd=home)
                    root_created = True
                except OSError as error:
                    raise SkillPathError("cannot create the skills root") from error
                root, root_state = _open_directory(
                    home, _SKILLS_DIRECTORY, required=True
                )
                assert root is not None
            assert root_state is not None
            _require_directory_identity(home, _SKILLS_DIRECTORY, root_state)

            directory, directory_state = _open_directory(
                root, skill.name, required=False
            )
            prior_state: os.stat_result | None = None
            if directory is None:
                if skill.name in by_name:
                    raise SkillPathError("skill directory identity changed")
                try:
                    os.mkdir(skill.name, mode=0o700, dir_fd=root)
                    directory_created = True
                except OSError as error:
                    raise SkillPathError("cannot create the skill directory") from error
                directory, directory_state = _open_directory(
                    root, skill.name, required=True
                )
                assert directory is not None
            else:
                prior_skill, prior_bytes, prior_state = _read_skill_document(
                    directory, skill.name
                )
                if by_name.get(skill.name) != prior_skill:
                    raise SkillPathError("skill content changed during save")
                if prior_bytes == rendered:
                    return False
            assert directory_state is not None

            temporary = f".{_SKILL_DOCUMENT}.{uuid4().hex}.tmp"
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            try:
                descriptor = os.open(temporary, flags, 0o600, dir_fd=directory)
            except OSError as error:
                raise SkillPathError(
                    "cannot create the temporary skill file"
                ) from error
            try:
                opened = os.fstat(descriptor)
                if not stat.S_ISREG(opened.st_mode) or opened.st_nlink != 1:
                    raise SkillPathError("temporary skill file is not private")
                with os.fdopen(descriptor, "wb") as file:
                    file.write(rendered)
                    file.flush()
                    os.fsync(file.fileno())
                descriptor = -1
            finally:
                if descriptor >= 0:
                    os.close(descriptor)

            _require_directory_identity(home, _SKILLS_DIRECTORY, root_state)
            _require_directory_identity(root, skill.name, directory_state)
            _require_unchanged_target(directory, _SKILL_DOCUMENT, prior_state)
            try:
                os.replace(
                    temporary,
                    _SKILL_DOCUMENT,
                    src_dir_fd=directory,
                    dst_dir_fd=directory,
                )
                temporary = None
                replaced = True
                os.fsync(directory)
            except OSError as error:
                raise SkillPathError("cannot atomically replace SKILL.md") from error
            written, written_bytes, _ = _read_skill_document(directory, skill.name)
            if written != skill or written_bytes != rendered:
                raise SkillStoreError("skill replacement did not round-trip exactly")
            _require_directory_identity(home, _SKILLS_DIRECTORY, root_state)
            _require_directory_identity(root, skill.name, directory_state)
            return True
        finally:
            if temporary is not None and directory is not None:
                try:
                    os.unlink(temporary, dir_fd=directory)
                except FileNotFoundError:
                    pass
            if directory is not None:
                os.close(directory)
            if directory_created and not replaced and root is not None:
                try:
                    os.rmdir(skill.name, dir_fd=root)
                except OSError:
                    pass
            if root is not None:
                os.close(root)
            if root_created and not replaced:
                try:
                    os.rmdir(_SKILLS_DIRECTORY, dir_fd=home)
                except OSError:
                    pass
            os.close(home)

    def _delete_sync(self, name: str) -> bool:
        home = self._open_home()
        try:
            root, root_state = _open_directory(home, _SKILLS_DIRECTORY, required=False)
            if root is None:
                return False
            try:
                current = _list_from_root(root)
                by_name = {item.name: item for item in current}
                if name not in by_name:
                    return False
                render_skill_index(
                    item.summary for item in current if item.name != name
                )
                directory, directory_state = _open_directory(root, name, required=True)
                assert directory is not None and directory_state is not None
                try:
                    skill, _, document_state = _read_skill_document(directory, name)
                    if skill != by_name[name]:
                        raise SkillPathError("skill content changed during delete")
                    _require_directory_identity(home, _SKILLS_DIRECTORY, root_state)
                    _require_directory_identity(root, name, directory_state)
                    _require_unchanged_target(
                        directory, _SKILL_DOCUMENT, document_state
                    )
                    try:
                        os.unlink(_SKILL_DOCUMENT, dir_fd=directory)
                        os.fsync(directory)
                    except OSError as error:
                        raise SkillPathError("cannot remove SKILL.md") from error
                finally:
                    os.close(directory)
                try:
                    os.rmdir(name, dir_fd=root)
                    os.fsync(root)
                except OSError as error:
                    raise SkillPathError("cannot remove the skill directory") from error
                _require_directory_identity(home, _SKILLS_DIRECTORY, root_state)
                return True
            finally:
                os.close(root)
        finally:
            os.close(home)

    def _open_home(self) -> int:
        try:
            lexical = os.lstat(self._home)
            if (
                not stat.S_ISDIR(lexical.st_mode)
                or stat.S_ISLNK(lexical.st_mode)
                or (lexical.st_dev, lexical.st_ino) != self._home_identity
                or self._home.resolve(strict=True) != self._home
            ):
                raise SkillPathError("agent home path identity changed")
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(self._home, flags)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != self._home_identity
            ):
                os.close(descriptor)
                raise SkillPathError("agent home directory identity changed")
            return descriptor
        except SkillPathError:
            raise
        except OSError as error:
            raise SkillPathError("agent home path is invalid") from error


def validate_skill_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("skill name must be text")
    if _SKILL_NAME.fullmatch(name) is None:
        raise SkillValidationError("skill name must match [a-z][a-z0-9-]{0,63}")


def render_skill_index(summaries: Iterable[SkillSummary]) -> str:
    values = tuple(sorted(tuple(summaries), key=lambda item: item.name))
    if len(values) > SKILL_MAX_COUNT:
        raise SkillValidationError(
            f"skill count exceeds the {SKILL_MAX_COUNT} skill limit"
        )
    if len({item.name for item in values}) != len(values):
        raise SkillValidationError("skill index contains duplicate names")
    text = "".join(f"- {item.name}: {item.description}\n" for item in values)
    if len(text) > SKILL_INDEX_MAX_CHARACTERS:
        raise SkillValidationError(
            f"skill index exceeds the {SKILL_INDEX_MAX_CHARACTERS} character limit"
        )
    data = text.encode("utf-8")
    if len(data) > SKILL_INDEX_MAX_UTF8_BYTES:
        raise SkillValidationError(
            f"skill index exceeds the {SKILL_INDEX_MAX_UTF8_BYTES} UTF-8 byte limit"
        )
    return text


def _validate_description(description: str) -> None:
    if not isinstance(description, str):
        raise TypeError("skill description must be text")
    if not description or description != description.strip():
        raise SkillValidationError(
            "skill description must be non-empty and already trimmed"
        )
    if len(description) > SKILL_DESCRIPTION_MAX_CHARACTERS:
        raise SkillValidationError("skill description exceeds the 240 character limit")
    if any(value in description for value in ("\r", "\n", "\0")):
        raise SkillValidationError("skill description must be one LF-free line")


def _validate_instructions(instructions: str) -> None:
    if not isinstance(instructions, str):
        raise TypeError("skill instructions must be text")
    if not instructions or instructions != instructions.strip():
        raise SkillValidationError(
            "skill instructions must be non-empty and already trimmed"
        )
    if len(instructions) > SKILL_INSTRUCTIONS_MAX_CHARACTERS:
        raise SkillValidationError(
            "skill instructions exceed the 12000 character limit"
        )
    if "\r" in instructions or "\0" in instructions:
        raise SkillValidationError("skill instructions must be LF-only without NUL")
    if any(line == "## Instructions" for line in instructions.split("\n")):
        raise SkillValidationError("skill instructions contain the reserved heading")


def _render_skill(skill: Skill) -> bytes:
    text = (
        f"# {skill.name}\n\n{skill.description}\n\n"
        f"## Instructions\n\n{skill.instructions}\n"
    )
    data = text.encode("utf-8")
    if len(data) > SKILL_RENDERED_MAX_UTF8_BYTES:
        raise SkillValidationError(
            "rendered SKILL.md exceeds the 50000 UTF-8 byte limit"
        )
    return data


def _rendered_document_sha256(data: bytes) -> str:
    """Return the one digest used by skill view and replacement preflight."""

    return sha256(data).hexdigest()


def _parse_skill(data: bytes, expected_name: str) -> Skill:
    if len(data) > SKILL_RENDERED_MAX_UTF8_BYTES:
        raise SkillValidationError("SKILL.md exceeds the 50000 UTF-8 byte limit")
    try:
        text = data.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise SkillValidationError("SKILL.md is not strict UTF-8") from error
    prefix = f"# {expected_name}\n\n"
    marker = "\n\n## Instructions\n\n"
    if not text.startswith(prefix) or not text.endswith("\n"):
        raise SkillValidationError("SKILL.md does not use the exact document grammar")
    remainder = text[len(prefix) : -1]
    if remainder.count(marker) != 1:
        raise SkillValidationError("SKILL.md must contain one reserved heading")
    description, instructions = remainder.split(marker)
    skill = Skill(expected_name, description, instructions)
    if _render_skill(skill) != data:
        raise SkillValidationError("SKILL.md is not rendered exactly")
    return skill


def _list_from_root(root: int) -> tuple[Skill, ...]:
    try:
        names = tuple(sorted(os.listdir(root)))
    except OSError as error:
        raise SkillPathError("cannot list the skills root") from error
    if len(names) > SKILL_MAX_COUNT:
        raise SkillValidationError(
            f"skill count exceeds the {SKILL_MAX_COUNT} skill limit"
        )
    skills: list[Skill] = []
    for name in names:
        validate_skill_name(name)
        directory, directory_state = _open_directory(root, name, required=True)
        assert directory is not None and directory_state is not None
        try:
            skills.append(_read_skill_directory(directory, name))
        finally:
            os.close(directory)
            _require_directory_identity(root, name, directory_state)
    render_skill_index(item.summary for item in skills)
    return tuple(skills)


def _read_skill_directory(directory: int, name: str) -> Skill:
    skill, _, _ = _read_skill_document(directory, name)
    return skill


def _read_skill_document(
    directory: int, name: str
) -> tuple[Skill, bytes, os.stat_result]:
    try:
        entries = tuple(os.listdir(directory))
    except OSError as error:
        raise SkillPathError("cannot list the skill directory") from error
    if entries != (_SKILL_DOCUMENT,):
        raise SkillPathError("a skill directory must contain exactly SKILL.md")
    state = _target_state(directory, _SKILL_DOCUMENT)
    if state is None:
        raise SkillPathError("SKILL.md is missing")
    _require_regular_owned_file(state, _SKILL_DOCUMENT)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(_SKILL_DOCUMENT, flags, dir_fd=directory)
    except OSError as error:
        raise SkillPathError("cannot open SKILL.md") from error
    try:
        opened = os.fstat(descriptor)
        _require_same_file_state(state, opened, _SKILL_DOCUMENT)
        with os.fdopen(descriptor, "rb") as file:
            data = file.read(SKILL_RENDERED_MAX_UTF8_BYTES + 1)
            final = os.fstat(file.fileno())
        descriptor = -1
        _require_same_file_state(state, final, _SKILL_DOCUMENT)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return _parse_skill(data, name), data, state


def _open_directory(
    parent: int,
    name: str,
    *,
    required: bool,
) -> tuple[int | None, os.stat_result | None]:
    state = _target_state(parent, name)
    if state is None:
        if required:
            raise SkillPathError(f"owned directory is missing: {name}")
        return None, None
    if not stat.S_ISDIR(state.st_mode) or stat.S_ISLNK(state.st_mode):
        raise SkillPathError(f"owned path is not a non-symlink directory: {name}")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=parent)
    except OSError as error:
        raise SkillPathError(f"cannot open owned directory: {name}") from error
    opened = os.fstat(descriptor)
    if not stat.S_ISDIR(opened.st_mode) or (opened.st_dev, opened.st_ino) != (
        state.st_dev,
        state.st_ino,
    ):
        os.close(descriptor)
        raise SkillPathError(f"owned directory identity changed: {name}")
    return descriptor, state


def _target_state(directory: int, name: str) -> os.stat_result | None:
    try:
        return os.stat(name, dir_fd=directory, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise SkillPathError(f"cannot inspect owned path: {name}") from error


def _require_directory_identity(
    parent: int,
    name: str,
    prior: os.stat_result | None,
) -> None:
    if prior is None:
        raise SkillPathError(f"owned directory identity is unavailable: {name}")
    current = _target_state(parent, name)
    if (
        current is None
        or not stat.S_ISDIR(current.st_mode)
        or stat.S_ISLNK(current.st_mode)
        or (current.st_dev, current.st_ino) != (prior.st_dev, prior.st_ino)
    ):
        raise SkillPathError(f"owned directory identity changed: {name}")


def _require_regular_owned_file(state: os.stat_result, name: str) -> None:
    if (
        not stat.S_ISREG(state.st_mode)
        or stat.S_ISLNK(state.st_mode)
        or state.st_nlink != 1
    ):
        raise SkillPathError(f"owned file must be regular and singly linked: {name}")


def _require_same_file_state(
    prior: os.stat_result,
    current: os.stat_result,
    name: str,
) -> None:
    _require_regular_owned_file(prior, name)
    _require_regular_owned_file(current, name)
    attributes = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size", "st_mtime_ns")
    if any(getattr(prior, item) != getattr(current, item) for item in attributes):
        raise SkillPathError(f"owned file identity changed: {name}")


def _state_fingerprint(*states: os.stat_result) -> str:
    values = tuple(
        (
            state.st_dev,
            state.st_ino,
            state.st_mode,
            state.st_nlink,
            state.st_size,
            state.st_mtime_ns,
            state.st_ctime_ns,
        )
        for state in states
    )
    return sha256(repr(values).encode("ascii")).hexdigest()


def _require_unchanged_target(
    directory: int,
    name: str,
    prior: os.stat_result | None,
) -> None:
    current = _target_state(directory, name)
    if prior is None:
        if current is not None:
            raise SkillPathError(f"owned target appeared during mutation: {name}")
        return
    if current is None:
        raise SkillPathError(f"owned target disappeared during mutation: {name}")
    _require_same_file_state(prior, current, name)


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
    "SKILL_DESCRIPTION_MAX_CHARACTERS",
    "SKILL_INDEX_MAX_CHARACTERS",
    "SKILL_INDEX_MAX_UTF8_BYTES",
    "SKILL_INSTRUCTIONS_MAX_CHARACTERS",
    "SKILL_MAX_COUNT",
    "SKILL_RENDERED_MAX_UTF8_BYTES",
    "Skill",
    "SkillNotFoundError",
    "SkillPathError",
    "SkillStore",
    "SkillStoreError",
    "SkillSummary",
    "SkillValidationError",
    "render_skill_index",
    "validate_skill_name",
]
