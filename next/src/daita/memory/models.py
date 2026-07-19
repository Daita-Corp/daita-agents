"""Portable immutable records for scoped, inspectable agent memory."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from hashlib import sha256
import math
import re
import unicodedata

from .._json import FrozenJsonObject, canonical_json

_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_IDENTITY_CHARACTERS = 512
_MAX_CONTENT_CHARACTERS = 32_000
_MAX_ATTRIBUTES_CHARACTERS = 32_000


def _required_text(
    value: str,
    field_name: str,
    *,
    maximum: int = _MAX_IDENTITY_CHARACTERS,
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")


def _optional_text(
    value: str | None,
    field_name: str,
    *,
    maximum: int = _MAX_IDENTITY_CHARACTERS,
) -> None:
    if value is not None:
        _required_text(value, field_name, maximum=maximum)


def _aware(value: datetime, field_name: str) -> None:
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or value.utcoffset() is None
    ):
        raise ValueError(f"{field_name} must be timezone-aware")


def _positive_int(value: int, field_name: str, *, maximum: int | None = None) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{field_name} must be a positive integer")
    if maximum is not None and value > maximum:
        raise ValueError(f"{field_name} exceeds {maximum}")


def _sensitivity_tuple(
    values: Iterable[MemorySensitivity],
) -> tuple[MemorySensitivity, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("allowed sensitivities must be a sequence")
    result = tuple(values)
    if not result:
        raise ValueError("allowed sensitivities cannot be empty")
    if any(not isinstance(value, MemorySensitivity) for value in result):
        raise TypeError("allowed sensitivities must contain MemorySensitivity values")
    if len(result) != len(set(result)):
        raise ValueError("allowed sensitivities cannot contain duplicates")
    return result


class MemoryKind(str, Enum):
    SEMANTIC_FACT = "semantic_fact"
    BUSINESS_DEFINITION = "business_definition"
    RESOURCE_ALIAS = "resource_alias"
    USER_PREFERENCE = "user_preference"
    EPISODIC_LESSON = "episodic_lesson"
    PROCEDURE_REFERENCE = "procedure_reference"


class MemoryCreator(str, Enum):
    USER = "user"
    AGENT = "agent"
    LEARNING_SERVICE = "learning_service"
    IMPORT = "import"


class MemorySensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


class MemoryState(str, Enum):
    ACTIVE = "active"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


class MemoryProvenanceKind(str, Enum):
    USER_STATEMENT = "user_statement"
    ACCEPTED_EVIDENCE = "accepted_evidence"
    IMPORT = "import"


class MemoryQualification(str, Enum):
    CURRENT = "current"
    UNBOUND = "unbound"
    EXPIRED = "expired"
    STALE_REVISION = "stale_revision"
    REVISION_UNKNOWN = "revision_unknown"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"


def normalize_memory_logical_key(value: str) -> str:
    """Return one stable correction identity without changing its meaning."""

    if not isinstance(value, str):
        raise TypeError("memory logical key must be a string")
    normalized = " ".join(unicodedata.normalize("NFKC", value).casefold().split())
    _required_text(normalized, "memory logical key")
    return normalized


@dataclass(frozen=True, slots=True)
class MemoryScope:
    """Hierarchical scope; unset dimensions are intentionally broader."""

    agent_id: str
    user_id: str | None = None
    session_id: str | None = None
    source_id: str | None = None
    resource_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "memory scope agent_id")
        for value, name in (
            (self.user_id, "memory scope user_id"),
            (self.session_id, "memory scope session_id"),
            (self.source_id, "memory scope source_id"),
            (self.resource_id, "memory scope resource_id"),
        ):
            _optional_text(value, name)
        if self.resource_id is not None and self.source_id is None:
            raise ValueError("resource-scoped memory requires a source_id")

    @property
    def fingerprint(self) -> str:
        material = canonical_json(
            {
                "agent_id": self.agent_id,
                "resource_id": self.resource_id,
                "session_id": self.session_id,
                "source_id": self.source_id,
                "user_id": self.user_id,
            }
        )
        return "sha256:" + sha256(material.encode("utf-8")).hexdigest()

    def contains(self, requested: MemoryScope) -> bool:
        """Whether this stored scope may be recalled in ``requested``."""

        if not isinstance(requested, MemoryScope):
            raise TypeError("requested scope must be a MemoryScope")
        if self.agent_id != requested.agent_id:
            return False
        for stored, current in (
            (self.user_id, requested.user_id),
            (self.session_id, requested.session_id),
            (self.source_id, requested.source_id),
            (self.resource_id, requested.resource_id),
        ):
            if stored is not None and stored != current:
                return False
        return True


@dataclass(frozen=True, slots=True)
class MemoryProvenance:
    kind: MemoryProvenanceKind
    content_hash: str
    operation_id: str | None = None
    trigger_id: str | None = None
    evidence_id: str | None = None
    session_id: str | None = None
    external_ref: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MemoryProvenanceKind):
            raise TypeError("memory provenance kind must be a MemoryProvenanceKind")
        if (
            not isinstance(self.content_hash, str)
            or _SHA256.fullmatch(self.content_hash) is None
        ):
            raise ValueError("memory provenance content_hash must use lowercase sha256")
        for value, name in (
            (self.operation_id, "memory provenance operation_id"),
            (self.trigger_id, "memory provenance trigger_id"),
            (self.evidence_id, "memory provenance evidence_id"),
            (self.session_id, "memory provenance session_id"),
            (self.external_ref, "memory provenance external_ref"),
        ):
            _optional_text(value, name)

        if self.kind is MemoryProvenanceKind.USER_STATEMENT:
            if self.operation_id is None or self.trigger_id is None:
                raise ValueError(
                    "user-statement provenance requires operation_id and trigger_id"
                )
            if self.evidence_id is not None or self.external_ref is not None:
                raise ValueError(
                    "user-statement provenance cannot reference evidence or an import"
                )
        elif self.kind is MemoryProvenanceKind.ACCEPTED_EVIDENCE:
            if self.operation_id is None or self.evidence_id is None:
                raise ValueError(
                    "accepted-evidence provenance requires operation_id and evidence_id"
                )
            if self.trigger_id is not None or self.external_ref is not None:
                raise ValueError(
                    "accepted-evidence provenance cannot reference a trigger or import"
                )
        else:
            if self.external_ref is None:
                raise ValueError("import provenance requires external_ref")
            if any(
                value is not None
                for value in (self.operation_id, self.trigger_id, self.evidence_id)
            ):
                raise ValueError("import provenance cannot reference runtime records")


@dataclass(frozen=True, slots=True)
class MemoryVersion:
    memory_id: str
    version: int
    content: str
    creator: MemoryCreator
    confidence: float
    sensitivity: MemorySensitivity
    provenance: MemoryProvenance
    created_at: datetime
    attributes: Mapping[str, object] = field(default_factory=dict)
    expires_at: datetime | None = None
    resource_revision: str | None = None
    supersedes_version: int | None = None

    def __post_init__(self) -> None:
        _required_text(self.memory_id, "memory version memory_id")
        _positive_int(self.version, "memory version")
        _required_text(
            self.content,
            "memory version content",
            maximum=_MAX_CONTENT_CHARACTERS,
        )
        if not isinstance(self.creator, MemoryCreator):
            raise TypeError("memory version creator must be a MemoryCreator")
        if (
            not isinstance(self.confidence, (int, float))
            or isinstance(self.confidence, bool)
            or not math.isfinite(float(self.confidence))
            or not 0.0 <= float(self.confidence) <= 1.0
        ):
            raise ValueError("memory confidence must be a finite number from 0 to 1")
        if not isinstance(self.sensitivity, MemorySensitivity):
            raise TypeError("memory sensitivity must be a MemorySensitivity")
        if not isinstance(self.provenance, MemoryProvenance):
            raise TypeError("memory provenance must be a MemoryProvenance")
        _aware(self.created_at, "memory version created_at")
        if self.expires_at is not None:
            _aware(self.expires_at, "memory version expires_at")
            if self.expires_at <= self.created_at:
                raise ValueError("memory expires_at must follow created_at")
        _optional_text(self.resource_revision, "memory resource_revision")
        if self.supersedes_version is not None:
            _positive_int(self.supersedes_version, "memory supersedes_version")
            if self.supersedes_version >= self.version:
                raise ValueError("supersedes_version must precede the new version")
        elif self.version != 1:
            raise ValueError("memory versions after version 1 must supersede a version")

        attributes = FrozenJsonObject.from_mapping(self.attributes)
        if len(canonical_json(attributes)) > _MAX_ATTRIBUTES_CHARACTERS:
            raise ValueError(
                f"memory attributes exceed {_MAX_ATTRIBUTES_CHARACTERS} characters"
            )
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "attributes", attributes)


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    id: str
    scope: MemoryScope
    kind: MemoryKind
    logical_key: str
    current_version: int
    state: MemoryState
    created_at: datetime
    updated_at: datetime
    superseded_by_id: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.id, "memory record id")
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("memory record scope must be a MemoryScope")
        if not isinstance(self.kind, MemoryKind):
            raise TypeError("memory record kind must be a MemoryKind")
        normalized_key = normalize_memory_logical_key(self.logical_key)
        if self.logical_key != normalized_key:
            raise ValueError(
                "memory logical_key must already be normalized with "
                "normalize_memory_logical_key"
            )
        _positive_int(self.current_version, "memory current_version")
        if not isinstance(self.state, MemoryState):
            raise TypeError("memory record state must be a MemoryState")
        _aware(self.created_at, "memory record created_at")
        _aware(self.updated_at, "memory record updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("memory updated_at cannot precede created_at")
        _optional_text(self.superseded_by_id, "memory superseded_by_id")
        if self.state is MemoryState.SUPERSEDED:
            if self.superseded_by_id is None:
                raise ValueError("superseded memory requires superseded_by_id")
            if self.superseded_by_id == self.id:
                raise ValueError("memory cannot be superseded by itself")
        elif self.superseded_by_id is not None:
            raise ValueError("only superseded memory may name superseded_by_id")

    @property
    def logical_identity(self) -> str:
        material = canonical_json(
            {
                "kind": self.kind.value,
                "logical_key": self.logical_key,
                "scope": self.scope.fingerprint,
            }
        )
        return "sha256:" + sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class MemorySnapshot:
    record: MemoryRecord
    version: MemoryVersion

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise TypeError("memory snapshot record must be a MemoryRecord")
        if not isinstance(self.version, MemoryVersion):
            raise TypeError("memory snapshot version must be a MemoryVersion")
        if self.record.id != self.version.memory_id:
            raise ValueError("memory snapshot record and version IDs do not match")
        if self.record.current_version != self.version.version:
            raise ValueError("memory snapshot must contain the current version")
        if (
            self.version.resource_revision is not None
            and self.record.scope.resource_id is None
        ):
            raise ValueError("revision-bound memory requires a resource-scoped record")


@dataclass(frozen=True, slots=True)
class MemoryHistory:
    record: MemoryRecord
    versions: tuple[MemoryVersion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.record, MemoryRecord):
            raise TypeError("memory history record must be a MemoryRecord")
        versions = tuple(self.versions)
        if not versions:
            raise ValueError("memory history requires at least one version")
        if any(not isinstance(value, MemoryVersion) for value in versions):
            raise TypeError("memory history must contain MemoryVersion records")
        if any(value.memory_id != self.record.id for value in versions):
            raise ValueError("memory history versions must belong to its record")
        numbers = [value.version for value in versions]
        if len(numbers) != len(set(numbers)):
            raise ValueError("memory history cannot contain duplicate versions")
        if self.record.current_version not in numbers:
            raise ValueError("memory history must include the current version")
        object.__setattr__(
            self,
            "versions",
            tuple(sorted(versions, key=lambda value: value.version)),
        )

    @property
    def current(self) -> MemoryVersion:
        return next(
            value
            for value in self.versions
            if value.version == self.record.current_version
        )


@dataclass(frozen=True, slots=True)
class MemoryRecallRequest:
    query: str
    scope: MemoryScope
    allowed_sensitivities: tuple[MemorySensitivity, ...] = (
        MemorySensitivity.PUBLIC,
        MemorySensitivity.INTERNAL,
    )
    current_resource_revision: str | None = None
    limit: int = 5
    character_budget: int = 4_000

    def __post_init__(self) -> None:
        _required_text(self.query, "memory recall query", maximum=4_096)
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("memory recall scope must be a MemoryScope")
        sensitivities = _sensitivity_tuple(self.allowed_sensitivities)
        _optional_text(
            self.current_resource_revision,
            "memory recall current_resource_revision",
        )
        if (
            self.current_resource_revision is not None
            and self.scope.resource_id is None
        ):
            raise ValueError(
                "current_resource_revision requires a resource-scoped recall"
            )
        _positive_int(self.limit, "memory recall limit", maximum=50)
        _positive_int(
            self.character_budget,
            "memory recall character_budget",
            maximum=32_000,
        )
        object.__setattr__(self, "allowed_sensitivities", sensitivities)

    @property
    def candidate_limit(self) -> int:
        return min(256, max(32, self.limit * 16))


@dataclass(frozen=True, slots=True)
class MemoryListRequest:
    scope: MemoryScope
    allowed_sensitivities: tuple[MemorySensitivity, ...] = (
        MemorySensitivity.PUBLIC,
        MemorySensitivity.INTERNAL,
    )
    current_resource_revision: str | None = None
    include_superseded: bool = False
    include_rejected: bool = False
    limit: int = 100

    def __post_init__(self) -> None:
        if not isinstance(self.scope, MemoryScope):
            raise TypeError("memory list scope must be a MemoryScope")
        sensitivities = _sensitivity_tuple(self.allowed_sensitivities)
        _optional_text(
            self.current_resource_revision,
            "memory list current_resource_revision",
        )
        if (
            self.current_resource_revision is not None
            and self.scope.resource_id is None
        ):
            raise ValueError(
                "current_resource_revision requires a resource-scoped list"
            )
        for value, name in (
            (self.include_superseded, "include_superseded"),
            (self.include_rejected, "include_rejected"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"memory list {name} must be a boolean")
        _positive_int(self.limit, "memory list limit", maximum=200)
        object.__setattr__(self, "allowed_sensitivities", sensitivities)


@dataclass(frozen=True, slots=True)
class MemoryInspectionRequest:
    agent_id: str
    memory_id: str
    current_resource_revision: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "memory inspection agent_id")
        _required_text(self.memory_id, "memory inspection memory_id")
        _optional_text(
            self.current_resource_revision,
            "memory inspection current_resource_revision",
        )


@dataclass(frozen=True, slots=True)
class MemorySupersessionRequest:
    agent_id: str
    memory_id: str
    expected_version: int
    replacement: MemoryVersion

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "memory supersession agent_id")
        _required_text(self.memory_id, "memory supersession memory_id")
        _positive_int(self.expected_version, "memory supersession expected_version")
        if not isinstance(self.replacement, MemoryVersion):
            raise TypeError("memory supersession replacement must be a MemoryVersion")
        if self.replacement.memory_id != self.memory_id:
            raise ValueError("replacement must belong to the superseded memory")
        if self.replacement.version != self.expected_version + 1:
            raise ValueError("replacement version must follow expected_version")
        if self.replacement.supersedes_version != self.expected_version:
            raise ValueError("replacement must supersede expected_version")


@dataclass(frozen=True, slots=True)
class MemoryRestoreRequest:
    agent_id: str
    memory_id: str
    expected_version: int
    restore_version: int
    replacement: MemoryVersion

    def __post_init__(self) -> None:
        _required_text(self.agent_id, "memory restore agent_id")
        _required_text(self.memory_id, "memory restore memory_id")
        _positive_int(self.expected_version, "memory restore expected_version")
        _positive_int(self.restore_version, "memory restore restore_version")
        if self.restore_version >= self.expected_version:
            raise ValueError("restore_version must precede the current version")
        if not isinstance(self.replacement, MemoryVersion):
            raise TypeError("memory restore replacement must be a MemoryVersion")
        if self.replacement.memory_id != self.memory_id:
            raise ValueError("restore replacement must belong to the memory")
        if self.replacement.version != self.expected_version + 1:
            raise ValueError("restore replacement must follow expected_version")
        if self.replacement.supersedes_version != self.expected_version:
            raise ValueError("restore replacement must supersede expected_version")


@dataclass(frozen=True, slots=True)
class QualifiedMemory:
    snapshot: MemorySnapshot
    qualification: MemoryQualification

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, MemorySnapshot):
            raise TypeError("qualified memory snapshot must be a MemorySnapshot")
        if not isinstance(self.qualification, MemoryQualification):
            raise TypeError(
                "qualified memory qualification must be a MemoryQualification"
            )


@dataclass(frozen=True, slots=True)
class MemoryRecallHit:
    memory: QualifiedMemory
    lexical_score: float

    def __post_init__(self) -> None:
        if not isinstance(self.memory, QualifiedMemory):
            raise TypeError("memory recall hit must contain QualifiedMemory")
        if (
            not isinstance(self.lexical_score, (int, float))
            or isinstance(self.lexical_score, bool)
            or not math.isfinite(float(self.lexical_score))
            or not 0.0 <= float(self.lexical_score) <= 1.0
        ):
            raise ValueError("memory lexical_score must be from 0 to 1")
        object.__setattr__(self, "lexical_score", float(self.lexical_score))


@dataclass(frozen=True, slots=True)
class MemoryRecallResult:
    hits: tuple[MemoryRecallHit, ...]
    candidate_count: int
    used_characters: int
    omitted_by_scope: int
    omitted_by_sensitivity: int
    omitted_by_lifecycle: int
    omitted_by_revision: int
    omitted_by_relevance: int
    omitted_by_budget: int
    omitted_by_limit: int
    truncated: bool

    def __post_init__(self) -> None:
        hits = tuple(self.hits)
        if any(not isinstance(value, MemoryRecallHit) for value in hits):
            raise TypeError("memory recall result must contain MemoryRecallHit records")
        for value, name in (
            (self.candidate_count, "candidate_count"),
            (self.used_characters, "used_characters"),
            (self.omitted_by_scope, "omitted_by_scope"),
            (self.omitted_by_sensitivity, "omitted_by_sensitivity"),
            (self.omitted_by_lifecycle, "omitted_by_lifecycle"),
            (self.omitted_by_revision, "omitted_by_revision"),
            (self.omitted_by_relevance, "omitted_by_relevance"),
            (self.omitted_by_budget, "omitted_by_budget"),
            (self.omitted_by_limit, "omitted_by_limit"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"memory recall {name} must be non-negative")
        if not isinstance(self.truncated, bool):
            raise TypeError("memory recall truncated must be a boolean")
        object.__setattr__(self, "hits", hits)


@dataclass(frozen=True, slots=True)
class MemoryListResult:
    items: tuple[QualifiedMemory, ...]
    candidate_count: int
    truncated: bool

    def __post_init__(self) -> None:
        items = tuple(self.items)
        if any(not isinstance(value, QualifiedMemory) for value in items):
            raise TypeError("memory list result must contain QualifiedMemory records")
        if (
            not isinstance(self.candidate_count, int)
            or isinstance(self.candidate_count, bool)
            or self.candidate_count < 0
        ):
            raise ValueError("memory list candidate_count must be non-negative")
        if not isinstance(self.truncated, bool):
            raise TypeError("memory list truncated must be a boolean")
        object.__setattr__(self, "items", items)


@dataclass(frozen=True, slots=True)
class MemoryInspection:
    history: MemoryHistory
    qualification: MemoryQualification

    def __post_init__(self) -> None:
        if not isinstance(self.history, MemoryHistory):
            raise TypeError("memory inspection history must be a MemoryHistory")
        if not isinstance(self.qualification, MemoryQualification):
            raise TypeError(
                "memory inspection qualification must be a MemoryQualification"
            )


__all__ = [
    "MemoryCreator",
    "MemoryHistory",
    "MemoryInspection",
    "MemoryInspectionRequest",
    "MemoryKind",
    "MemoryListRequest",
    "MemoryListResult",
    "MemoryProvenance",
    "MemoryProvenanceKind",
    "MemoryQualification",
    "MemoryRecallHit",
    "MemoryRecallRequest",
    "MemoryRecallResult",
    "MemoryRecord",
    "MemoryRestoreRequest",
    "MemoryScope",
    "MemorySensitivity",
    "MemorySnapshot",
    "MemoryState",
    "MemorySupersessionRequest",
    "MemoryVersion",
    "QualifiedMemory",
    "normalize_memory_logical_key",
]
