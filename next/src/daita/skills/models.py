"""Portable immutable records for procedural skill discovery and activation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
import re

_STABLE_NAME = re.compile(r"[a-z][a-z0-9]*(?:-[a-z0-9]+)*\Z")
_VERSION = re.compile(
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)\."
    r"(?:0|[1-9][0-9]*)"
    r"(?:-[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?"
    r"(?:\+[0-9A-Za-z]+(?:[.-][0-9A-Za-z]+)*)?\Z"
)
_CAPABILITY_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")
_CLASSIFIER = re.compile(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*\Z")
_CANONICAL_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")


def _required_text(value: str, field_name: str, *, maximum: int) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip() or value != value.strip() or "\x00" in value:
        raise ValueError(f"{field_name} must be a non-empty normalized string")
    if len(value) > maximum:
        raise ValueError(f"{field_name} must contain at most {maximum} characters")


def _optional_text(value: str | None, field_name: str, *, maximum: int) -> None:
    if value is not None:
        _required_text(value, field_name, maximum=maximum)


def _aware(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


def _stable_name(value: str, field_name: str = "skill stable_name") -> None:
    _required_text(value, field_name, maximum=64)
    if _STABLE_NAME.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be a lowercase hyphen-separated stable name"
        )


def _version(value: str) -> None:
    _required_text(value, "skill version", maximum=64)
    if _VERSION.fullmatch(value) is None:
        raise ValueError("skill version must be a canonical semantic version")


def _canonical_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or _CANONICAL_SHA256.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a canonical lowercase sha256 hash")


def _text_tuple(
    value: tuple[str, ...],
    field_name: str,
    *,
    maximum_items: int,
    item_pattern: re.Pattern[str],
    item_maximum: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    items = tuple(value)
    if len(items) > maximum_items:
        raise ValueError(f"{field_name} contains too many items")
    for item in items:
        _required_text(item, f"{field_name} item", maximum=item_maximum)
        if item_pattern.fullmatch(item) is None:
            raise ValueError(f"{field_name} contains an invalid item: {item!r}")
    if len(items) != len(set(items)):
        raise ValueError(f"{field_name} must not contain duplicates")
    if items != tuple(sorted(items)):
        raise ValueError(f"{field_name} must be in deterministic sorted order")
    return items


class SkillActivationMode(str, Enum):
    EXPLICIT = "explicit"
    ON_DEMAND = "on_demand"
    ALWAYS = "always"


class SkillSource(str, Enum):
    BUILTIN = "builtin"
    USER = "user"
    EXTENSION = "extension"
    LEARNED_PROPOSAL = "learned_proposal"


class SkillSelectionReason(str, Enum):
    EXPLICIT = "explicit"
    ON_DEMAND = "on_demand"
    ALWAYS = "always"


@dataclass(frozen=True, slots=True)
class Skill:
    """Stable skill identity; instructions live only on immutable versions."""

    id: str
    agent_id: str
    stable_name: str
    source: SkillSource
    created_at: datetime

    def __post_init__(self) -> None:
        _required_text(self.id, "skill id", maximum=256)
        _required_text(self.agent_id, "skill agent_id", maximum=256)
        _stable_name(self.stable_name)
        if not isinstance(self.source, SkillSource):
            raise TypeError("skill source must be a SkillSource")
        _aware(self.created_at, "skill created_at")


@dataclass(frozen=True, slots=True)
class SkillVersion:
    """One immutable metadata and instruction snapshot of a ``SKILL.md`` file."""

    id: str
    agent_id: str
    skill_id: str
    stable_name: str
    version: str
    description: str
    domains: tuple[str, ...]
    resource_kinds: tuple[str, ...]
    required_capability_ids: tuple[str, ...]
    activation_mode: SkillActivationMode
    sensitivity_notes: str | None
    policy_notes: str | None
    source: SkillSource
    content_hash: str
    instructions: str
    source_path: str | None
    created_at: datetime

    def __post_init__(self) -> None:
        _required_text(self.id, "skill version id", maximum=256)
        _required_text(self.agent_id, "skill version agent_id", maximum=256)
        _required_text(self.skill_id, "skill version skill_id", maximum=256)
        _stable_name(self.stable_name)
        _version(self.version)
        _required_text(self.description, "skill description", maximum=1_024)
        object.__setattr__(
            self,
            "domains",
            _text_tuple(
                self.domains,
                "skill domains",
                maximum_items=32,
                item_pattern=_CLASSIFIER,
                item_maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "resource_kinds",
            _text_tuple(
                self.resource_kinds,
                "skill resource_kinds",
                maximum_items=32,
                item_pattern=_CLASSIFIER,
                item_maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "required_capability_ids",
            _text_tuple(
                self.required_capability_ids,
                "skill required_capability_ids",
                maximum_items=64,
                item_pattern=_CAPABILITY_ID,
                item_maximum=128,
            ),
        )
        if not isinstance(self.activation_mode, SkillActivationMode):
            raise TypeError("skill activation_mode must be a SkillActivationMode")
        _optional_text(
            self.sensitivity_notes,
            "skill sensitivity_notes",
            maximum=2_048,
        )
        _optional_text(self.policy_notes, "skill policy_notes", maximum=2_048)
        if not isinstance(self.source, SkillSource):
            raise TypeError("skill version source must be a SkillSource")
        _canonical_hash(self.content_hash, "skill content_hash")
        _required_text(self.instructions, "skill instructions", maximum=65_536)
        if self.source_path is not None:
            _required_text(self.source_path, "skill source_path", maximum=256)
            source_path = PurePosixPath(self.source_path)
            if (
                source_path.is_absolute()
                or source_path.as_posix() != self.source_path
                or ".." in source_path.parts
                or self.source_path != f"{self.stable_name}/SKILL.md"
            ):
                raise ValueError("skill source_path must be <stable-name>/SKILL.md")
        _aware(self.created_at, "skill version created_at")


@dataclass(frozen=True, slots=True)
class SkillIndex:
    """Compact SQLite-friendly projection that excludes full instructions."""

    agent_id: str
    skill_id: str
    version_id: str
    stable_name: str
    version: str
    description: str
    domains: tuple[str, ...]
    resource_kinds: tuple[str, ...]
    required_capability_ids: tuple[str, ...]
    activation_mode: SkillActivationMode
    source: SkillSource
    content_hash: str
    active_version_id: str | None
    updated_at: datetime

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "skill index agent_id", maximum=256)
        _required_text(self.skill_id, "skill index skill_id", maximum=256)
        _required_text(self.version_id, "skill index version_id", maximum=256)
        _stable_name(self.stable_name)
        _version(self.version)
        _required_text(self.description, "skill index description", maximum=1_024)
        object.__setattr__(
            self,
            "domains",
            _text_tuple(
                self.domains,
                "skill index domains",
                maximum_items=32,
                item_pattern=_CLASSIFIER,
                item_maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "resource_kinds",
            _text_tuple(
                self.resource_kinds,
                "skill index resource_kinds",
                maximum_items=32,
                item_pattern=_CLASSIFIER,
                item_maximum=64,
            ),
        )
        object.__setattr__(
            self,
            "required_capability_ids",
            _text_tuple(
                self.required_capability_ids,
                "skill index required_capability_ids",
                maximum_items=64,
                item_pattern=_CAPABILITY_ID,
                item_maximum=128,
            ),
        )
        if not isinstance(self.activation_mode, SkillActivationMode):
            raise TypeError("skill index activation_mode must be a SkillActivationMode")
        if not isinstance(self.source, SkillSource):
            raise TypeError("skill index source must be a SkillSource")
        _canonical_hash(self.content_hash, "skill index content_hash")
        _optional_text(
            self.active_version_id,
            "skill index active_version_id",
            maximum=256,
        )
        _aware(self.updated_at, "skill index updated_at")

    @classmethod
    def from_version(
        cls,
        version: SkillVersion,
        *,
        active_version_id: str | None = None,
        updated_at: datetime | None = None,
    ) -> SkillIndex:
        if not isinstance(version, SkillVersion):
            raise TypeError("version must be a SkillVersion")
        resolved_updated_at = version.created_at if updated_at is None else updated_at
        _aware(resolved_updated_at, "skill index updated_at")
        return cls(
            agent_id=version.agent_id,
            skill_id=version.skill_id,
            version_id=version.id,
            stable_name=version.stable_name,
            version=version.version,
            description=version.description,
            domains=version.domains,
            resource_kinds=version.resource_kinds,
            required_capability_ids=version.required_capability_ids,
            activation_mode=version.activation_mode,
            source=version.source,
            content_hash=version.content_hash,
            active_version_id=active_version_id,
            updated_at=resolved_updated_at,
        )

    def matches(self, version: SkillVersion) -> bool:
        return isinstance(version, SkillVersion) and (
            self.agent_id,
            self.skill_id,
            self.version_id,
            self.stable_name,
            self.version,
            self.description,
            self.domains,
            self.resource_kinds,
            self.required_capability_ids,
            self.activation_mode,
            self.source,
            self.content_hash,
        ) == (
            version.agent_id,
            version.skill_id,
            version.id,
            version.stable_name,
            version.version,
            version.description,
            version.domains,
            version.resource_kinds,
            version.required_capability_ids,
            version.activation_mode,
            version.source,
            version.content_hash,
        )


@dataclass(frozen=True, slots=True)
class SkillActivation:
    """One append-only explicit change of a skill's active version pointer."""

    id: str
    agent_id: str
    skill_id: str
    version_id: str
    previous_version_id: str | None
    actor_id: str
    reason: str
    activated_at: datetime

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.id, "skill activation id"),
            (self.agent_id, "skill activation agent_id"),
            (self.skill_id, "skill activation skill_id"),
            (self.version_id, "skill activation version_id"),
            (self.actor_id, "skill activation actor_id"),
        ):
            _required_text(value, field_name, maximum=256)
        _optional_text(
            self.previous_version_id,
            "skill activation previous_version_id",
            maximum=256,
        )
        if self.previous_version_id == self.version_id:
            raise ValueError("skill activation must change the active version")
        _required_text(self.reason, "skill activation reason", maximum=1_024)
        _aware(self.activated_at, "skill activation activated_at")


@dataclass(frozen=True, slots=True)
class SkillInspection:
    """Complete bounded audit view for one stable skill."""

    skill: Skill
    index: SkillIndex
    versions: tuple[SkillVersion, ...]
    activations: tuple[SkillActivation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.skill, Skill):
            raise TypeError("skill inspection skill must be a Skill")
        if not isinstance(self.index, SkillIndex):
            raise TypeError("skill inspection index must be a SkillIndex")
        versions = tuple(self.versions)
        activations = tuple(self.activations)
        if any(not isinstance(item, SkillVersion) for item in versions):
            raise TypeError("skill inspection versions must contain SkillVersion")
        if any(not isinstance(item, SkillActivation) for item in activations):
            raise TypeError("skill inspection activations must contain SkillActivation")
        if (
            self.index.agent_id != self.skill.agent_id
            or self.index.skill_id != self.skill.id
        ):
            raise ValueError("skill inspection index belongs to another skill")
        if any(
            item.agent_id != self.skill.agent_id or item.skill_id != self.skill.id
            for item in versions
        ):
            raise ValueError("skill inspection version belongs to another skill")
        if any(
            item.agent_id != self.skill.agent_id or item.skill_id != self.skill.id
            for item in activations
        ):
            raise ValueError("skill inspection activation belongs to another skill")
        if len({item.id for item in versions}) != len(versions):
            raise ValueError("skill inspection has duplicate version identities")
        if len({item.id for item in activations}) != len(activations):
            raise ValueError("skill inspection has duplicate activation identities")
        version_ids = {item.id for item in versions}
        if self.index.version_id not in version_ids:
            raise ValueError("skill index version is absent from inspection history")
        if (
            self.index.active_version_id is not None
            and self.index.active_version_id not in version_ids
        ):
            raise ValueError("active skill version is absent from inspection history")
        if any(item.version_id not in version_ids for item in activations):
            raise ValueError("skill activation references an unknown version")
        if tuple((item.activated_at, item.id) for item in activations) != tuple(
            sorted((item.activated_at, item.id) for item in activations)
        ):
            raise ValueError("skill activations must be in deterministic time order")
        if activations:
            if self.index.active_version_id != activations[-1].version_id:
                raise ValueError("latest activation must match the active version")
        elif self.index.active_version_id is not None:
            raise ValueError("active skill version requires activation history")
        object.__setattr__(self, "versions", versions)
        object.__setattr__(self, "activations", activations)


@dataclass(frozen=True, slots=True)
class SkillSelection:
    """A selected active version with full instructions loaded on demand."""

    index: SkillIndex
    version: SkillVersion
    reason: SkillSelectionReason

    def __post_init__(self) -> None:
        if not isinstance(self.index, SkillIndex):
            raise TypeError("skill selection index must be a SkillIndex")
        if not isinstance(self.version, SkillVersion):
            raise TypeError("skill selection version must be a SkillVersion")
        if not isinstance(self.reason, SkillSelectionReason):
            raise TypeError("skill selection reason must be a SkillSelectionReason")
        if self.index.active_version_id != self.version.id:
            raise ValueError("skill selection must load the active version")
        if not self.index.matches(self.version):
            raise ValueError("skill selection index and version metadata disagree")


__all__ = [
    "Skill",
    "SkillActivation",
    "SkillActivationMode",
    "SkillIndex",
    "SkillInspection",
    "SkillSelection",
    "SkillSelectionReason",
    "SkillSource",
    "SkillVersion",
]
