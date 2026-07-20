"""Bounded ``SKILL.md`` discovery, activation, selection, and inspection."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import os
from pathlib import Path
import re
import stat
import tomllib
from typing import Protocol
from uuid import uuid4

from ..errors import SkillError as PublicSkillError
from .models import (
    Skill,
    SkillActivation,
    SkillActivationMode,
    SkillIndex,
    SkillInspection,
    SkillSelection,
    SkillSelectionReason,
    SkillSource,
    SkillVersion,
)

_STABLE_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_CAPABILITY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_WORD = re.compile(r"[A-Za-z0-9]+")
_ALLOWED_METADATA = frozenset(
    {
        "activation_mode",
        "description",
        "domains",
        "name",
        "policy_notes",
        "required_capability_ids",
        "resource_kinds",
        "sensitivity_notes",
        "version",
    }
)
_FORBIDDEN_METADATA = frozenset(
    {
        "capabilities",
        "capability",
        "code",
        "executor",
        "executor_id",
        "executors",
        "hidden_tools",
        "policy",
        "policy_ids",
        "policies",
        "provides_capabilities",
        "python",
        "runtime_effects",
        "tool",
        "tool_views",
        "tools",
        "worker",
        "workers",
    }
)
_DEFAULT_MAX_SKILL_BYTES = 64 * 1_024
_MAX_CONFIGURED_SKILL_BYTES = 1 * 1_024 * 1_024
_DEFAULT_MAX_SKILLS = 128
_MAX_CONFIGURED_SKILLS = 1_024
_DEFAULT_SELECTION_LIMIT = 8
_MAX_SELECTION_LIMIT = 32
_DEFAULT_SELECTION_CHARACTERS = 64 * 1_024
_MAX_SELECTION_CHARACTERS = 1 * 1_024 * 1_024


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class SkillError(PublicSkillError):
    """Base class for portable skill-service failures."""


class SkillDiscoveryError(SkillError):
    """Raised when a configured skill root is not a stable bounded tree."""


class SkillFormatError(SkillError):
    """Raised when one ``SKILL.md`` file violates the strict portable format."""


class SkillNotFoundError(SkillError):
    """Raised when an exact skill or version identity is unknown."""


class SkillNotActiveError(SkillError):
    """Raised when explicit selection targets a discovered but inactive skill."""


class SkillCapabilityUnavailableError(SkillError):
    """Raised when explicit activation or selection lacks required capabilities."""


class SkillActivationConflictError(SkillError):
    """Raised when an activation's expected current version is stale."""


class SkillSelectionBudgetError(SkillError):
    """Raised when an explicitly requested skill cannot fit its bounded budget."""


class SkillStore(Protocol):
    """Narrow lifecycle repository; implementations own atomic persistence."""

    async def record_discovery(
        self,
        skill: Skill,
        version: SkillVersion,
        index: SkillIndex,
    ) -> SkillIndex: ...

    async def list_skill_index(self, agent_id: str) -> tuple[SkillIndex, ...]: ...

    async def load_skill_index(
        self,
        agent_id: str,
        skill_id: str,
    ) -> SkillIndex | None: ...

    async def load_skill_version(
        self,
        agent_id: str,
        version_id: str,
    ) -> SkillVersion | None: ...

    async def inspect_skill(
        self,
        agent_id: str,
        skill_id: str,
    ) -> SkillInspection | None: ...

    async def activate_skill(
        self,
        activation: SkillActivation,
        *,
        expected_active_version_id: str | None,
    ) -> SkillInspection: ...


class SkillService:
    """Coordinate inert procedure files without owning runtime effects."""

    def __init__(
        self,
        *,
        agent_id: str,
        root: str | Path,
        source: SkillSource,
        store: SkillStore,
        capability_ids: frozenset[str],
        max_skill_bytes: int = _DEFAULT_MAX_SKILL_BYTES,
        max_skills: int = _DEFAULT_MAX_SKILLS,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
    ) -> None:
        _required_identity(agent_id, "skill-service agent_id")
        if not isinstance(root, (str, Path)):
            raise TypeError("skill root must be a string or Path")
        if not isinstance(source, SkillSource):
            raise TypeError("skill source must be a SkillSource")
        for method_name in (
            "record_discovery",
            "list_skill_index",
            "load_skill_index",
            "load_skill_version",
            "inspect_skill",
            "activate_skill",
        ):
            if not callable(getattr(store, method_name, None)):
                raise TypeError(f"skill store must provide {method_name}")
        if not isinstance(capability_ids, frozenset):
            raise TypeError("capability_ids must be a read-only frozenset")
        for capability_id in capability_ids:
            if (
                not isinstance(capability_id, str)
                or _CAPABILITY_ID.fullmatch(capability_id) is None
            ):
                raise ValueError("capability_ids contains an invalid capability ID")
        _bounded_integer(
            max_skill_bytes,
            "max_skill_bytes",
            maximum=_MAX_CONFIGURED_SKILL_BYTES,
        )
        _bounded_integer(
            max_skills,
            "max_skills",
            maximum=_MAX_CONFIGURED_SKILLS,
        )
        if not callable(clock):
            raise TypeError("skill clock must be callable")
        if not callable(id_factory):
            raise TypeError("skill id_factory must be callable")

        self._agent_id = agent_id
        self._root = Path(root)
        self._source = source
        self._store = store
        self._capability_ids = capability_ids
        self._max_skill_bytes = max_skill_bytes
        self._max_skills = max_skills
        self._clock = clock
        self._id_factory = id_factory

    async def refresh(self) -> tuple[SkillIndex, ...]:
        """Discover and index a complete validated snapshot of the configured root."""

        created_at = self._clock()
        _aware(created_at, "skill discovery time")
        discovered = await asyncio.to_thread(
            _discover_skill_root,
            self._root,
            agent_id=self._agent_id,
            source=self._source,
            created_at=created_at,
            max_skill_bytes=self._max_skill_bytes,
            max_skills=self._max_skills,
        )
        committed: list[SkillIndex] = []
        for skill, version, index in discovered:
            stored = await self._store.record_discovery(skill, version, index)
            if not isinstance(stored, SkillIndex):
                raise TypeError("skill store record_discovery must return SkillIndex")
            if stored.agent_id != self._agent_id or stored.skill_id != skill.id:
                raise ValueError("skill store returned an index for another skill")
            committed.append(stored)
        return tuple(
            sorted(committed, key=lambda item: (item.stable_name, item.skill_id))
        )

    async def list(self) -> tuple[SkillIndex, ...]:
        values = tuple(await self._store.list_skill_index(self._agent_id))
        if any(not isinstance(item, SkillIndex) for item in values):
            raise TypeError("skill store index contains a non-SkillIndex record")
        if any(item.agent_id != self._agent_id for item in values):
            raise ValueError("skill store index contains another agent's skill")
        if len({item.skill_id for item in values}) != len(values):
            raise ValueError("skill store index contains duplicate skill identities")
        return tuple(sorted(values, key=lambda item: (item.stable_name, item.skill_id)))

    async def inspect(self, skill_id: str) -> SkillInspection:
        _required_identity(skill_id, "skill_id")
        inspection = await self._store.inspect_skill(self._agent_id, skill_id)
        if inspection is None:
            raise SkillNotFoundError(f"unknown skill: {skill_id}")
        if not isinstance(inspection, SkillInspection):
            raise TypeError(
                "skill store inspect_skill must return SkillInspection or None"
            )
        if inspection.skill.agent_id != self._agent_id:
            raise ValueError("skill store returned another agent's inspection")
        return inspection

    async def prepare_change_activation(
        self,
        skill: Skill,
        version: SkillVersion,
        *,
        expected_active_version_id: str | None,
        actor_id: str,
        reason: str,
    ) -> SkillActivation:
        """Validate and build an activation for an atomically staged change.

        This method performs no persistence.  It exists so a learning-owned
        transaction can reuse the skill owner's scope, capability, optimistic
        pointer, and audit-record validation before committing a new inert
        version and its explicit activation together.
        """

        if not isinstance(skill, Skill):
            raise TypeError("skill change skill must be a Skill")
        if not isinstance(version, SkillVersion):
            raise TypeError("skill change version must be a SkillVersion")
        if (
            skill.agent_id != self._agent_id
            or version.agent_id != self._agent_id
            or version.skill_id != skill.id
            or version.stable_name != skill.stable_name
            or version.source is not skill.source
        ):
            raise ValueError("skill change records belong to another identity")
        if version.created_at < skill.created_at:
            raise ValueError("skill change version cannot precede its skill")
        current = await self._store.load_skill_index(self._agent_id, skill.id)
        if current is not None and (
            not isinstance(current, SkillIndex)
            or current.agent_id != self._agent_id
            or current.skill_id != skill.id
            or current.stable_name != skill.stable_name
            or current.source is not skill.source
        ):
            raise ValueError("skill store returned another skill change identity")
        return self._prepare_activation(
            version,
            current,
            expected_active_version_id=expected_active_version_id,
            actor_id=actor_id,
            reason=reason,
        )

    async def activate(
        self,
        skill_id: str,
        version_id: str,
        *,
        expected_active_version_id: str | None,
        actor_id: str,
        reason: str,
    ) -> SkillInspection:
        """Explicitly activate one known version through an optimistic store commit."""

        _required_identity(skill_id, "skill_id")
        _required_identity(version_id, "skill version_id")
        _optional_identity(
            expected_active_version_id,
            "expected_active_version_id",
        )
        _required_identity(actor_id, "skill activation actor_id")
        _required_bounded_text(reason, "skill activation reason", maximum=1_024)
        index = await self._store.load_skill_index(self._agent_id, skill_id)
        if index is None:
            raise SkillNotFoundError(f"unknown skill: {skill_id}")
        if not isinstance(index, SkillIndex):
            raise TypeError("skill store load_skill_index returned an invalid record")
        if index.agent_id != self._agent_id or index.skill_id != skill_id:
            raise ValueError("skill store returned an index for another skill")
        version = await self._store.load_skill_version(self._agent_id, version_id)
        if version is None or version.skill_id != skill_id:
            raise SkillNotFoundError(f"unknown skill version: {version_id}")
        if version.agent_id != self._agent_id:
            raise ValueError("skill store returned another agent's version")
        activation = self._prepare_activation(
            version,
            index,
            expected_active_version_id=expected_active_version_id,
            actor_id=actor_id,
            reason=reason,
        )
        inspection = await self._store.activate_skill(
            activation,
            expected_active_version_id=expected_active_version_id,
        )
        if not isinstance(inspection, SkillInspection):
            raise TypeError("skill store activate_skill must return SkillInspection")
        if inspection.index.active_version_id != version_id:
            raise ValueError("skill store did not commit the requested active version")
        return inspection

    def _prepare_activation(
        self,
        version: SkillVersion,
        current: SkillIndex | None,
        *,
        expected_active_version_id: str | None,
        actor_id: str,
        reason: str,
    ) -> SkillActivation:
        _optional_identity(
            expected_active_version_id,
            "expected_active_version_id",
        )
        _required_identity(actor_id, "skill activation actor_id")
        _required_bounded_text(reason, "skill activation reason", maximum=1_024)
        if version.agent_id != self._agent_id:
            raise ValueError("skill version belongs to another agent")
        if current is not None and (
            current.agent_id != self._agent_id
            or current.skill_id != version.skill_id
            or current.stable_name != version.stable_name
            or current.source is not version.source
        ):
            raise ValueError("skill index and version identities disagree")
        active_version_id = None if current is None else current.active_version_id
        if active_version_id != expected_active_version_id:
            raise SkillActivationConflictError(
                f"skill {version.skill_id} active version changed before activation"
            )
        if active_version_id == version.id:
            raise SkillActivationConflictError(
                f"skill {version.skill_id} version is already active"
            )
        self._require_capabilities(
            version.required_capability_ids,
            version.skill_id,
        )
        activated_at = self._clock()
        _aware(activated_at, "skill activation time")
        return SkillActivation(
            id=self._id_factory("skill-activation"),
            agent_id=self._agent_id,
            skill_id=version.skill_id,
            version_id=version.id,
            previous_version_id=expected_active_version_id,
            actor_id=actor_id,
            reason=reason,
            activated_at=activated_at,
        )

    async def select(
        self,
        query: str,
        *,
        explicit_skill_ids: Sequence[str] = (),
        limit: int = _DEFAULT_SELECTION_LIMIT,
        max_instruction_characters: int = _DEFAULT_SELECTION_CHARACTERS,
    ) -> tuple[SkillSelection, ...]:
        """Load only active deterministic selections within a hard context budget."""

        _required_bounded_text(query, "skill selection query", maximum=4_096)
        if isinstance(explicit_skill_ids, (str, bytes)):
            raise TypeError("explicit_skill_ids must be a sequence of skill IDs")
        explicit = tuple(explicit_skill_ids)
        for skill_id in explicit:
            _required_identity(skill_id, "explicit skill_id")
        if len(explicit) != len(set(explicit)):
            raise ValueError("explicit_skill_ids must not contain duplicates")
        if len(explicit) > _MAX_SELECTION_LIMIT:
            raise ValueError("explicit_skill_ids contains too many skills")
        _bounded_integer(limit, "skill selection limit", maximum=_MAX_SELECTION_LIMIT)
        _bounded_integer(
            max_instruction_characters,
            "max_instruction_characters",
            maximum=_MAX_SELECTION_CHARACTERS,
        )

        index = await self.list()
        by_id = {item.skill_id: item for item in index}
        unknown = tuple(sorted(set(explicit) - set(by_id)))
        if unknown:
            raise SkillNotFoundError("unknown explicit skill(s): " + ", ".join(unknown))

        candidates: list[tuple[int, int, str, SkillIndex, SkillSelectionReason]] = []
        explicit_set = set(explicit)
        for item in index:
            if item.skill_id in explicit_set:
                if item.active_version_id is None:
                    raise SkillNotActiveError(f"skill is not active: {item.skill_id}")
                self._require_capabilities(
                    item.required_capability_ids,
                    item.skill_id,
                )
                candidates.append(
                    (0, 0, item.skill_id, item, SkillSelectionReason.EXPLICIT)
                )
                continue
            if item.active_version_id is None:
                continue
            if not set(item.required_capability_ids) <= self._capability_ids:
                continue
            if item.activation_mode is SkillActivationMode.ALWAYS:
                candidates.append(
                    (1, 0, item.skill_id, item, SkillSelectionReason.ALWAYS)
                )
                continue
            if item.activation_mode is SkillActivationMode.ON_DEMAND:
                score = _selection_score(query, item)
                if score > 0:
                    candidates.append(
                        (
                            2,
                            -score,
                            item.skill_id,
                            item,
                            SkillSelectionReason.ON_DEMAND,
                        )
                    )

        candidates.sort(key=lambda item: item[:3])
        selected: list[SkillSelection] = []
        used_characters = 0
        for _, _, _, item, selection_reason in candidates:
            if len(selected) >= limit:
                break
            assert item.active_version_id is not None
            version = await self._store.load_skill_version(
                self._agent_id,
                item.active_version_id,
            )
            if version is None:
                raise ValueError("active skill version is missing from the skill store")
            selection = SkillSelection(
                index=item,
                version=version,
                reason=selection_reason,
            )
            next_total = used_characters + len(version.instructions)
            if next_total > max_instruction_characters:
                if selection_reason is SkillSelectionReason.EXPLICIT:
                    raise SkillSelectionBudgetError(
                        f"explicit skill does not fit the instruction budget: "
                        f"{item.skill_id}"
                    )
                continue
            selected.append(selection)
            used_characters = next_total
        return tuple(selected)

    def _require_capabilities(
        self,
        required: tuple[str, ...],
        skill_id: str,
    ) -> None:
        missing = tuple(sorted(set(required) - self._capability_ids))
        if missing:
            raise SkillCapabilityUnavailableError(
                f"skill {skill_id} requires unavailable capabilities: "
                + ", ".join(missing)
            )


@dataclass(frozen=True, slots=True)
class _RootDescriptor:
    path: Path
    descriptor: int
    device: int
    inode: int


def _discover_skill_root(
    root_path: Path,
    *,
    agent_id: str,
    source: SkillSource,
    created_at: datetime,
    max_skill_bytes: int,
    max_skills: int,
) -> tuple[tuple[Skill, SkillVersion, SkillIndex], ...]:
    root = _open_root(root_path)
    try:
        names = tuple(sorted(os.listdir(root.descriptor)))
        if len(names) > max_skills:
            raise SkillDiscoveryError(
                f"skill root contains more than {max_skills} entries"
            )
        discovered: list[tuple[Skill, SkillVersion, SkillIndex]] = []
        for name in names:
            if _STABLE_NAME.fullmatch(name) is None:
                raise SkillDiscoveryError(
                    "skill root entries must be lowercase stable-name directories"
                )
            before = _lstat_at(root.descriptor, name, "skill directory")
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise SkillDiscoveryError(
                    f"skill root entry is not a real directory: {name}"
                )
            directory = _open_directory_at(root.descriptor, name, before)
            try:
                entries = tuple(sorted(os.listdir(directory)))
                if entries != ("SKILL.md",):
                    raise SkillDiscoveryError(
                        f"skill directory must contain only SKILL.md: {name}"
                    )
                raw = _read_skill_file(directory, max_skill_bytes, name)
                current_directory = _lstat_at(
                    root.descriptor,
                    name,
                    "skill directory",
                )
                opened_directory = os.fstat(directory)
                if not _same_identity(current_directory, opened_directory):
                    raise SkillDiscoveryError(
                        f"skill directory changed during discovery: {name}"
                    )
            finally:
                os.close(directory)
            discovered.append(
                _parse_skill(
                    raw,
                    directory_name=name,
                    agent_id=agent_id,
                    source=source,
                    created_at=created_at,
                )
            )
        _verify_root(root)
        return tuple(discovered)
    except SkillError:
        raise
    except (OSError, ValueError) as error:
        raise SkillDiscoveryError("skill root could not be read safely") from error
    finally:
        os.close(root.descriptor)


def _parse_skill(
    raw: bytes,
    *,
    directory_name: str,
    agent_id: str,
    source: SkillSource,
    created_at: datetime,
) -> tuple[Skill, SkillVersion, SkillIndex]:
    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise SkillFormatError(
            f"skill must be strict UTF-8: {directory_name}"
        ) from error
    if "\x00" in text:
        raise SkillFormatError(f"skill contains a NUL character: {directory_name}")
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "+++":
        raise SkillFormatError(
            f"skill must begin with TOML front matter: {directory_name}"
        )
    closing_index = next(
        (
            index
            for index, line in enumerate(lines[1:], start=1)
            if line.rstrip("\r\n") == "+++"
        ),
        None,
    )
    if closing_index is None:
        raise SkillFormatError(
            f"skill TOML front matter is not closed: {directory_name}"
        )
    metadata_text = "".join(lines[1:closing_index])
    if len(metadata_text.encode("utf-8")) > 16 * 1_024:
        raise SkillFormatError(f"skill metadata is too large: {directory_name}")
    try:
        metadata = tomllib.loads(metadata_text)
    except tomllib.TOMLDecodeError as error:
        raise SkillFormatError(f"skill TOML is invalid: {directory_name}") from error
    unknown = set(metadata) - _ALLOWED_METADATA
    if unknown:
        forbidden = tuple(sorted(unknown & _FORBIDDEN_METADATA))
        if forbidden:
            raise SkillFormatError(
                "skill metadata cannot declare runtime effects: " + ", ".join(forbidden)
            )
        raise SkillFormatError(
            "skill metadata contains unknown fields: " + ", ".join(sorted(unknown))
        )
    required = {"activation_mode", "description", "name", "version"}
    missing = tuple(sorted(required - set(metadata)))
    if missing:
        raise SkillFormatError(
            "skill metadata is missing required fields: " + ", ".join(missing)
        )
    if any(isinstance(value, dict) for value in metadata.values()):
        raise SkillFormatError("skill metadata cannot contain nested tables")

    name = _metadata_text(metadata["name"], "name", maximum=64)
    if _STABLE_NAME.fullmatch(name) is None or name != directory_name:
        raise SkillFormatError(
            "skill name must exactly match its stable-name directory"
        )
    version_text = _metadata_text(metadata["version"], "version", maximum=64)
    description = _metadata_text(
        metadata["description"],
        "description",
        maximum=1_024,
    )
    try:
        activation_mode = SkillActivationMode(
            _metadata_text(
                metadata["activation_mode"],
                "activation_mode",
                maximum=32,
            )
        )
    except ValueError as error:
        raise SkillFormatError(
            "skill activation_mode must be explicit, on_demand, or always"
        ) from error
    domains = _metadata_string_array(metadata.get("domains", []), "domains", 32, 64)
    resource_kinds = _metadata_string_array(
        metadata.get("resource_kinds", []),
        "resource_kinds",
        32,
        64,
    )
    required_capabilities = _metadata_string_array(
        metadata.get("required_capability_ids", []),
        "required_capability_ids",
        64,
        128,
    )
    if any(_CAPABILITY_ID.fullmatch(value) is None for value in required_capabilities):
        raise SkillFormatError("required_capability_ids contains an invalid ID")
    sensitivity_notes = _metadata_optional_text(
        metadata.get("sensitivity_notes"),
        "sensitivity_notes",
        maximum=2_048,
    )
    policy_notes = _metadata_optional_text(
        metadata.get("policy_notes"),
        "policy_notes",
        maximum=2_048,
    )
    instructions = "".join(lines[closing_index + 1 :]).strip()
    if not instructions:
        raise SkillFormatError(f"skill instructions are empty: {directory_name}")
    if len(instructions) > 65_536:
        raise SkillFormatError(f"skill instructions are too large: {directory_name}")

    content_hash = "sha256:" + sha256(raw).hexdigest()
    skill_id = f"skill:{name}"
    version_id = f"skill-version:{content_hash[7:]}"
    try:
        skill = Skill(
            id=skill_id,
            agent_id=agent_id,
            stable_name=name,
            source=source,
            created_at=created_at,
        )
        skill_version = SkillVersion(
            id=version_id,
            agent_id=agent_id,
            skill_id=skill_id,
            stable_name=name,
            version=version_text,
            description=description,
            domains=domains,
            resource_kinds=resource_kinds,
            required_capability_ids=required_capabilities,
            activation_mode=activation_mode,
            sensitivity_notes=sensitivity_notes,
            policy_notes=policy_notes,
            source=source,
            content_hash=content_hash,
            instructions=instructions,
            source_path=f"{name}/SKILL.md",
            created_at=created_at,
        )
    except (TypeError, ValueError) as error:
        raise SkillFormatError(
            f"skill metadata violates the portable contract: {directory_name}"
        ) from error
    return skill, skill_version, SkillIndex.from_version(skill_version)


def _selection_score(query: str, index: SkillIndex) -> int:
    query_terms = set(_WORD.findall(query.casefold()))
    if not query_terms:
        return 0
    name_terms = set(_WORD.findall(index.stable_name.casefold()))
    description_terms = set(_WORD.findall(index.description.casefold()))
    domain_terms = {
        term
        for value in (*index.domains, *index.resource_kinds)
        for term in _WORD.findall(value.casefold())
    }
    return (
        8 * len(query_terms & name_terms)
        + 3 * len(query_terms & domain_terms)
        + len(query_terms & description_terms)
    )


def _root_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_root(value: Path) -> _RootDescriptor:
    raw = os.fspath(value)
    if (
        not isinstance(raw, str)
        or not raw
        or "\x00" in raw
        or not os.path.isabs(raw)
        or os.path.abspath(raw) != raw
    ):
        raise SkillDiscoveryError(
            "skill root must be an existing absolute canonical directory"
        )
    components = Path(raw).parts
    descriptors: list[int] = []
    try:
        descriptor = os.open(os.path.sep, _root_flags())
        descriptors.append(descriptor)
        for component in components[1:]:
            before = _lstat_at(descriptor, component, "skill root component")
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise OSError("skill root component is not a real directory")
            child = _open_directory_at(descriptor, component, before)
            descriptors.append(child)
            descriptor = child
        opened = os.fstat(descriptor)
        for ancestor in descriptors[:-1]:
            os.close(ancestor)
        return _RootDescriptor(
            path=Path(raw),
            descriptor=descriptor,
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
        )
    except (OSError, SkillDiscoveryError) as error:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        if isinstance(error, SkillDiscoveryError):
            raise
        raise SkillDiscoveryError(
            "skill root must be an existing absolute canonical directory"
        ) from error


def _verify_root(expected: _RootDescriptor) -> None:
    actual = _open_root(expected.path)
    try:
        if (actual.device, actual.inode) != (expected.device, expected.inode):
            raise SkillDiscoveryError("skill root changed during discovery")
    finally:
        os.close(actual.descriptor)


def _open_directory_at(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
) -> int:
    descriptor = os.open(name, _root_flags(), dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not _same_identity(before, opened):
            raise OSError("directory changed while opening")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_skill_file(directory: int, maximum: int, skill_name: str) -> bytes:
    before = _lstat_at(directory, "SKILL.md", "skill file")
    if (
        not stat.S_ISREG(before.st_mode)
        or stat.S_ISLNK(before.st_mode)
        or int(before.st_nlink) != 1
    ):
        raise SkillDiscoveryError(
            f"SKILL.md must be one unaliased regular file: {skill_name}"
        )
    if before.st_size > maximum:
        raise SkillDiscoveryError(f"SKILL.md exceeds its byte limit: {skill_name}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open("SKILL.md", flags, dir_fd=directory)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or int(opened.st_nlink) != 1
            or not _same_version(before, opened)
        ):
            raise SkillDiscoveryError(f"SKILL.md changed while opening: {skill_name}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum:
                raise SkillDiscoveryError(
                    f"SKILL.md exceeds its byte limit: {skill_name}"
                )
        finished = os.fstat(descriptor)
        current = _lstat_at(directory, "SKILL.md", "skill file")
        if not _same_version(opened, finished) or not _same_version(finished, current):
            raise SkillDiscoveryError(f"SKILL.md changed during read: {skill_name}")
        raw = b"".join(chunks)
        if len(raw) != int(finished.st_size):
            raise SkillDiscoveryError(f"SKILL.md read was incomplete: {skill_name}")
        return raw
    finally:
        os.close(descriptor)


def _lstat_at(directory: int, name: str, label: str) -> os.stat_result:
    try:
        return os.stat(name, dir_fd=directory, follow_symlinks=False)
    except OSError as error:
        raise SkillDiscoveryError(f"cannot inspect {label}") from error


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (int(left.st_dev), int(left.st_ino)) == (
        int(right.st_dev),
        int(right.st_ino),
    )


def _same_version(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        int(left.st_dev),
        int(left.st_ino),
        int(left.st_mode),
        int(left.st_nlink),
        int(left.st_size),
        int(left.st_mtime_ns),
        int(left.st_ctime_ns),
    ) == (
        int(right.st_dev),
        int(right.st_ino),
        int(right.st_mode),
        int(right.st_nlink),
        int(right.st_size),
        int(right.st_mtime_ns),
        int(right.st_ctime_ns),
    )


def _metadata_text(value: object, name: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise SkillFormatError(f"skill {name} must be a bounded normalized string")
    return value


def _metadata_optional_text(
    value: object,
    name: str,
    *,
    maximum: int,
) -> str | None:
    if value is None:
        return None
    return _metadata_text(value, name, maximum=maximum)


def _metadata_string_array(
    value: object,
    name: str,
    maximum_items: int,
    maximum_characters: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) > maximum_items:
        raise SkillFormatError(f"skill {name} must be a bounded string array")
    values = tuple(
        _metadata_text(item, f"{name} item", maximum=maximum_characters)
        for item in value
    )
    if len(values) != len(set(values)):
        raise SkillFormatError(f"skill {name} must not contain duplicates")
    return tuple(sorted(values))


def _required_identity(value: str, field_name: str) -> None:
    _required_bounded_text(value, field_name, maximum=256)


def _optional_identity(value: str | None, field_name: str) -> None:
    if value is not None:
        _required_identity(value, field_name)


def _required_bounded_text(value: str, field_name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or "\x00" in value
        or len(value) > maximum
    ):
        raise ValueError(f"{field_name} must be a bounded normalized string")


def _bounded_integer(value: int, field_name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{field_name} must be from 1 through {maximum}")


def _aware(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


__all__ = [
    "SkillActivationConflictError",
    "SkillCapabilityUnavailableError",
    "SkillDiscoveryError",
    "SkillError",
    "SkillFormatError",
    "SkillNotActiveError",
    "SkillNotFoundError",
    "SkillSelectionBudgetError",
    "SkillService",
    "SkillStore",
]
