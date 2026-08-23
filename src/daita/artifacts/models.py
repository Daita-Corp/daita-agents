"""Define and validate artifact drafts, references, provenance, and delivery records."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path

from .._json import FrozenJsonObject
from ..catalog.models import Sensitivity
from ..errors import DaitaError, ErrorRetryability

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_DOCUMENT_BYTES = 256 * 1024
MAX_DOCUMENT_CHARACTERS = 200_000
MAX_ARTIFACTS_PER_RUN = 8
MAX_ARTIFACT_BYTES_PER_RUN = 128 * 1024 * 1024
MAX_ARTIFACTS_PER_AGENT = 10_000
MAX_ARTIFACT_BYTES_PER_AGENT = 2 * 1024 * 1024 * 1024
MAX_PERSISTENT_DESTINATIONS = 32
MAX_ONE_TIME_DESTINATIONS = 8
MAX_FILENAME_UTF8_BYTES = 120
MAX_DELIVERY_FILENAME_UTF8_BYTES = 128
MAX_COLLISION_SUFFIX = 9_999

SYSTEM_DOWNLOADS_DESTINATION_ID = "destination-system-downloads"
DEFAULT_DESTINATION_SELECTOR = "default"

_ARTIFACT_ID = re.compile(r"artifact-[0-9a-f]{32}\Z")
_DESTINATION_ID = re.compile(r"destination-[0-9a-f]{32}\Z")
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_WINDOWS_DEVICE_STEMS = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)
_RESERVED_FILENAME_CHARACTERS = frozenset('<>:"|?*')


def _required_text(value: str, name: str, *, maximum: int = 2_048) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    if len(value) > maximum:
        raise ValueError(f"{name} exceeds {maximum} characters")
    if any(character in "\r\n\x00" for character in value):
        raise ValueError(f"{name} must be single-line text")


def _digest(value: str, name: str) -> None:
    if not isinstance(value, str) or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical sha256 digest")


def _utc(value: datetime, name: str) -> None:
    offset = value.utcoffset() if isinstance(value, datetime) else None
    if (
        not isinstance(value, datetime)
        or value.tzinfo is None
        or offset is None
        or offset.total_seconds() != 0
    ):
        raise ValueError(f"{name} must be timezone-aware UTC")


def _safe_single_line(value: str, name: str, *, maximum: int) -> None:
    _required_text(value, name, maximum=maximum)
    if any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in value
    ):
        raise ValueError(f"{name} contains unsafe Unicode controls")


class ArtifactError(DaitaError):
    """One safe, structured artifact failure family."""

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        self.code = code
        self.message = message
        self.details = FrozenJsonObject.from_mapping(details or {})
        super().__init__(
            message,
            error_code=code,
            retryability=ErrorRetryability.PERMANENT,
        )


class ArtifactAuthorship(str, Enum):
    EXACT_SOURCE_DATA = "exact_source_data"
    MODEL_AUTHORED_ANALYSIS = "model_authored_analysis"


@dataclass(frozen=True, slots=True)
class ArtifactResourceBinding:
    source_id: str
    source_revision: str
    resource_id: str
    resource_revision: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "artifact source_id"),
            (self.source_revision, "artifact source_revision"),
            (self.resource_id, "artifact resource_id"),
            (self.resource_revision, "artifact resource_revision"),
        ):
            _required_text(value, name, maximum=1_024)


@dataclass(frozen=True, slots=True)
class ArtifactProvenance:
    authorship: ArtifactAuthorship
    evidence_call_ids: tuple[str, ...] = ()
    derived_from_artifact_id: str | None = None
    resource_bindings: tuple[ArtifactResourceBinding, ...] = ()
    sql_fingerprint: str | None = None
    parameters_sha256: str | None = None
    columns: tuple[str, ...] = ()
    row_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.authorship, ArtifactAuthorship):
            raise TypeError("artifact authorship must be ArtifactAuthorship")
        evidence = tuple(self.evidence_call_ids)
        if len(evidence) > 16 or len(evidence) != len(set(evidence)):
            raise ValueError("artifact evidence_call_ids exceed their distinct bound")
        for item in evidence:
            _required_text(item, "artifact evidence call id", maximum=256)
        if self.derived_from_artifact_id is not None and (
            not isinstance(self.derived_from_artifact_id, str)
            or _ARTIFACT_ID.fullmatch(self.derived_from_artifact_id) is None
        ):
            raise ValueError(
                "artifact derived_from_artifact_id must use "
                "artifact-<32 lowercase hex>"
            )
        bindings = tuple(self.resource_bindings)
        if len(bindings) > 64 or any(
            not isinstance(item, ArtifactResourceBinding) for item in bindings
        ):
            raise ValueError("artifact resource bindings exceed their bound")
        if len({(item.source_id, item.resource_id) for item in bindings}) != len(
            bindings
        ):
            raise ValueError("artifact resource bindings cannot contain duplicates")
        if bindings != tuple(
            sorted(bindings, key=lambda item: (item.source_id, item.resource_id))
        ):
            raise ValueError("artifact resource bindings must be sorted")
        columns = tuple(self.columns)
        if len(columns) > 256:
            raise ValueError("artifact columns exceed 256 items")
        for item in columns:
            _required_text(item, "artifact column", maximum=256)
        if self.row_count is not None and (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or self.row_count < 0
        ):
            raise ValueError("artifact row_count must be non-negative or None")
        if self.sql_fingerprint is not None:
            _digest(self.sql_fingerprint, "artifact sql_fingerprint")
        if self.parameters_sha256 is not None:
            _digest(self.parameters_sha256, "artifact parameters_sha256")
        if self.authorship is ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS and (
            self.derived_from_artifact_id is not None
            or self.sql_fingerprint is not None
            or self.parameters_sha256 is not None
            or bool(columns)
            or self.row_count is not None
        ):
            raise ValueError(
                "model-authored provenance cannot claim exact export facts"
            )
        if self.authorship is ArtifactAuthorship.EXACT_SOURCE_DATA:
            has_query_facts = (
                self.sql_fingerprint is not None
                or self.parameters_sha256 is not None
                or bool(columns)
                or self.row_count is not None
            )
            query_facts_complete = (
                self.sql_fingerprint is not None
                and self.parameters_sha256 is not None
                and bool(columns)
                and self.row_count is not None
            )
            if (
                evidence
                or not 1 <= len(bindings) <= 64
                or (has_query_facts and not query_facts_complete)
            ):
                raise ValueError(
                    "exact-source provenance requires complete runtime facts"
                )
        object.__setattr__(self, "evidence_call_ids", evidence)
        object.__setattr__(self, "resource_bindings", bindings)
        object.__setattr__(self, "columns", columns)


@dataclass(frozen=True, slots=True)
class ArtifactDraft:
    content: bytes
    suggested_filename: str
    media_type: str
    sensitivity: Sensitivity
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        if not isinstance(self.content, bytes):
            raise TypeError("artifact draft content must be bytes")
        if not self.content:
            raise ValueError("artifact draft content must be non-empty")
        if len(self.content) > MAX_ARTIFACT_BYTES:
            raise ValueError("artifact draft exceeds the global byte bound")
        validate_safe_filename(self.suggested_filename)
        _required_text(self.media_type, "artifact media_type", maximum=128)
        if not isinstance(self.sensitivity, Sensitivity):
            raise TypeError("artifact sensitivity must be Sensitivity")
        if not isinstance(self.provenance, ArtifactProvenance):
            raise TypeError("artifact provenance must be ArtifactProvenance")


@dataclass(frozen=True, slots=True)
class ArtifactRef:
    artifact_id: str
    run_id: str
    conversation_id: str
    call_id: str
    capability_id: str
    filename: str
    media_type: str
    byte_size: int
    sha256: str
    sensitivity: Sensitivity
    provenance: ArtifactProvenance
    created_at: datetime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_id, str)
            or _ARTIFACT_ID.fullmatch(self.artifact_id) is None
        ):
            raise ValueError("artifact_id must use artifact-<32 lowercase hex>")
        for value, name in (
            (self.run_id, "artifact run_id"),
            (self.conversation_id, "artifact conversation_id"),
            (self.call_id, "artifact call_id"),
            (self.capability_id, "artifact capability_id"),
        ):
            _required_text(value, name, maximum=256)
        validate_safe_filename(self.filename)
        _required_text(self.media_type, "artifact media_type", maximum=128)
        if (
            not isinstance(self.byte_size, int)
            or isinstance(self.byte_size, bool)
            or not 1 <= self.byte_size <= MAX_ARTIFACT_BYTES
        ):
            raise ValueError("artifact byte_size is outside the global bound")
        _digest(self.sha256, "artifact sha256")
        if (
            not isinstance(self.sensitivity, Sensitivity)
            or self.sensitivity is Sensitivity.UNKNOWN
        ):
            raise ValueError("artifact sensitivity must be a resolved sensitivity")
        if not isinstance(self.provenance, ArtifactProvenance):
            raise TypeError("artifact provenance must be ArtifactProvenance")
        _utc(self.created_at, "artifact created_at")


@dataclass(frozen=True, slots=True)
class ArtifactPayload:
    ref: ArtifactRef
    content: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.ref, ArtifactRef):
            raise TypeError("artifact payload ref must be ArtifactRef")
        if not isinstance(self.content, bytes):
            raise TypeError("artifact payload content must be bytes")
        if len(self.content) != self.ref.byte_size:
            raise ValueError("artifact payload size does not match its ref")


class ArtifactDestinationKind(str, Enum):
    SYSTEM_DOWNLOADS = "system_downloads"
    LOCAL_DIRECTORY = "local_directory"


class DestinationAuthorization(str, Enum):
    SYSTEM = "system"
    ONE_TIME = "one_time"
    PERSISTENT = "persistent"


class DestinationAvailability(str, Enum):
    AVAILABLE = "available"
    AUTHORIZATION_REQUIRED = "authorization_required"
    REVOKED = "revoked"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class ArtifactDestination:
    destination_id: str
    display_name: str
    kind: ArtifactDestinationKind
    authorization: DestinationAuthorization
    availability: DestinationAvailability
    is_default: bool

    def __post_init__(self) -> None:
        if self.destination_id != SYSTEM_DOWNLOADS_DESTINATION_ID and (
            not isinstance(self.destination_id, str)
            or _DESTINATION_ID.fullmatch(self.destination_id) is None
        ):
            raise ValueError("destination_id has an invalid form")
        _safe_single_line(self.display_name, "destination display_name", maximum=80)
        if not isinstance(self.kind, ArtifactDestinationKind):
            raise TypeError("destination kind must be ArtifactDestinationKind")
        if not isinstance(self.authorization, DestinationAuthorization):
            raise TypeError("destination authorization is invalid")
        if not isinstance(self.availability, DestinationAvailability):
            raise TypeError("destination availability is invalid")
        if not isinstance(self.is_default, bool):
            raise TypeError("destination is_default must be a boolean")


@dataclass(frozen=True, slots=True)
class ArtifactDeliveryReceipt:
    artifact_id: str
    destination_id: str
    filename: str
    saved_path: str
    byte_size: int
    sha256: str
    renamed_for_collision: bool
    delivered_at: datetime

    def __post_init__(self) -> None:
        if (
            not isinstance(self.artifact_id, str)
            or _ARTIFACT_ID.fullmatch(self.artifact_id) is None
        ):
            raise ValueError("delivery artifact_id has an invalid form")
        if self.destination_id != SYSTEM_DOWNLOADS_DESTINATION_ID and (
            not isinstance(self.destination_id, str)
            or _DESTINATION_ID.fullmatch(self.destination_id) is None
        ):
            raise ValueError("delivery destination_id has an invalid form")
        validate_safe_filename(
            self.filename, maximum_bytes=MAX_DELIVERY_FILENAME_UTF8_BYTES
        )
        _safe_single_line(self.saved_path, "delivery saved_path", maximum=4_096)
        if not Path(self.saved_path).is_absolute():
            raise ValueError("delivery saved_path must be absolute")
        if (
            not isinstance(self.byte_size, int)
            or isinstance(self.byte_size, bool)
            or not 1 <= self.byte_size <= MAX_ARTIFACT_BYTES
        ):
            raise ValueError("delivery byte_size is outside the artifact bound")
        _digest(self.sha256, "delivery sha256")
        if not isinstance(self.renamed_for_collision, bool):
            raise TypeError("renamed_for_collision must be a boolean")
        _utc(self.delivered_at, "delivery delivered_at")


def validate_safe_filename(
    value: str,
    *,
    maximum_bytes: int = MAX_FILENAME_UTF8_BYTES,
) -> str:
    """Return the NFC basename or raise one safe structured filename error."""

    if not isinstance(value, str):
        raise ArtifactError(
            "artifact_unsafe_filename",
            "The artifact filename must be text.",
            {"reason": "not_text"},
        )
    normalized = unicodedata.normalize("NFC", value)
    reason: str | None = None
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError:
        encoded = b""
        reason = "invalid_unicode"
    if not normalized:
        reason = "empty"
    elif len(encoded) > maximum_bytes:
        reason = "too_long"
    elif "/" in normalized or "\\" in normalized or "\x00" in normalized:
        reason = "path_separator"
    elif any(
        unicodedata.category(character) in {"Cc", "Cf", "Cs"}
        for character in normalized
    ):
        reason = "unsafe_unicode"
    elif any(character in _RESERVED_FILENAME_CHARACTERS for character in normalized):
        reason = "reserved_character"
    elif normalized in {".", ".."}:
        reason = "reserved_basename"
    elif normalized != normalized.strip() or normalized.endswith("."):
        reason = "unsafe_boundary"
    elif normalized.split(".", 1)[0].casefold() in _WINDOWS_DEVICE_STEMS:
        reason = "reserved_device_name"
    if reason is not None:
        raise ArtifactError(
            "artifact_unsafe_filename",
            "The artifact filename is unsafe.",
            {"reason": reason},
        )
    return normalized


def canonical_artifact_filename(
    value: str,
    media_type: str,
    allowed_extensions: tuple[tuple[str, tuple[str, ...]], ...],
) -> str:
    normalized = validate_safe_filename(value)
    allowed = dict(allowed_extensions).get(media_type)
    if not allowed:
        raise ArtifactError(
            "artifact_invalid_format",
            "The artifact media type is not allowed.",
            {"media_type": media_type, "allowed_extensions": ()},
        )
    stem, dot, extension = normalized.rpartition(".")
    if not dot or not stem or "." in stem or f".{extension.casefold()}" not in allowed:
        raise ArtifactError(
            "artifact_invalid_format",
            "The artifact filename extension does not match its media type.",
            {"media_type": media_type, "allowed_extensions": allowed},
        )
    canonical = f"{stem}.{extension.casefold()}"
    return validate_safe_filename(canonical)


def _datetime_text(value: datetime) -> str:
    _utc(value, "serialized datetime")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _datetime_value(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{name} must be a canonical UTC timestamp")
    parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    _utc(parsed, name)
    return parsed


def _exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    name: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{name} has an invalid shape")


def _mapping_text(value: Mapping[str, object], key: str, name: str) -> str:
    selected = value.get(key)
    if not isinstance(selected, str):
        raise ValueError(f"{name} {key} must be text")
    return selected


def _mapping_int(value: Mapping[str, object], key: str, name: str) -> int:
    selected = value.get(key)
    if type(selected) is not int:
        raise ValueError(f"{name} {key} must be an integer")
    return selected


def artifact_provenance_to_mapping(value: ArtifactProvenance) -> dict[str, object]:
    return {
        "authorship": value.authorship.value,
        "evidence_call_ids": value.evidence_call_ids,
        "derived_from_artifact_id": value.derived_from_artifact_id,
        "resource_bindings": tuple(
            {
                "source_id": item.source_id,
                "source_revision": item.source_revision,
                "resource_id": item.resource_id,
                "resource_revision": item.resource_revision,
            }
            for item in value.resource_bindings
        ),
        "sql_fingerprint": value.sql_fingerprint,
        "parameters_sha256": value.parameters_sha256,
        "columns": value.columns,
        "row_count": value.row_count,
    }


def artifact_provenance_from_mapping(value: Mapping[str, object]) -> ArtifactProvenance:
    expected_keys = frozenset(
        {
            "authorship",
            "evidence_call_ids",
            "derived_from_artifact_id",
            "resource_bindings",
            "sql_fingerprint",
            "parameters_sha256",
            "columns",
            "row_count",
        }
    )
    actual_keys = set(value)
    if actual_keys not in (
        expected_keys,
        expected_keys - {"derived_from_artifact_id"},
    ):
        raise ValueError("artifact provenance has an invalid shape")
    raw_bindings = value.get("resource_bindings", ())
    if not isinstance(raw_bindings, (tuple, list)):
        raise ValueError("artifact resource_bindings must be an array")
    bindings = []
    for item in raw_bindings:
        if not isinstance(item, Mapping):
            raise ValueError("artifact resource binding must be an object")
        _exact_keys(
            item,
            frozenset(
                {
                    "source_id",
                    "source_revision",
                    "resource_id",
                    "resource_revision",
                }
            ),
            "artifact resource binding",
        )
        bindings.append(
            ArtifactResourceBinding(
                source_id=_mapping_text(item, "source_id", "artifact binding"),
                source_revision=_mapping_text(
                    item, "source_revision", "artifact binding"
                ),
                resource_id=_mapping_text(item, "resource_id", "artifact binding"),
                resource_revision=_mapping_text(
                    item, "resource_revision", "artifact binding"
                ),
            )
        )
    evidence = value.get("evidence_call_ids", ())
    columns = value.get("columns", ())
    if not isinstance(evidence, (tuple, list)) or not isinstance(
        columns, (tuple, list)
    ):
        raise ValueError("artifact provenance arrays are invalid")
    if any(not isinstance(item, str) for item in (*evidence, *columns)):
        raise ValueError("artifact provenance arrays must contain text")
    raw_row_count = value.get("row_count")
    if raw_row_count is not None and type(raw_row_count) is not int:
        raise ValueError("artifact provenance row_count is invalid")
    row_count = raw_row_count if type(raw_row_count) is int else None
    sql_fingerprint = value.get("sql_fingerprint")
    parameters_sha256 = value.get("parameters_sha256")
    derived_from_artifact_id = value.get("derived_from_artifact_id")
    if derived_from_artifact_id is not None and not isinstance(
        derived_from_artifact_id, str
    ):
        raise ValueError("artifact derived_from_artifact_id is invalid")
    if sql_fingerprint is not None and not isinstance(sql_fingerprint, str):
        raise ValueError("artifact provenance sql_fingerprint is invalid")
    if parameters_sha256 is not None and not isinstance(parameters_sha256, str):
        raise ValueError("artifact provenance parameters_sha256 is invalid")
    return ArtifactProvenance(
        authorship=ArtifactAuthorship(value.get("authorship")),
        evidence_call_ids=tuple(evidence),
        derived_from_artifact_id=derived_from_artifact_id,
        resource_bindings=tuple(bindings),
        sql_fingerprint=sql_fingerprint,
        parameters_sha256=parameters_sha256,
        columns=tuple(columns),
        row_count=row_count,
    )


def artifact_ref_to_mapping(value: ArtifactRef) -> dict[str, object]:
    return {
        "artifact_id": value.artifact_id,
        "run_id": value.run_id,
        "conversation_id": value.conversation_id,
        "call_id": value.call_id,
        "capability_id": value.capability_id,
        "filename": value.filename,
        "media_type": value.media_type,
        "byte_size": value.byte_size,
        "sha256": value.sha256,
        "sensitivity": value.sensitivity.value,
        "provenance": artifact_provenance_to_mapping(value.provenance),
        "created_at": _datetime_text(value.created_at),
    }


def artifact_ref_from_mapping(value: Mapping[str, object]) -> ArtifactRef:
    _exact_keys(
        value,
        frozenset(
            {
                "artifact_id",
                "run_id",
                "conversation_id",
                "call_id",
                "capability_id",
                "filename",
                "media_type",
                "byte_size",
                "sha256",
                "sensitivity",
                "provenance",
                "created_at",
            }
        ),
        "artifact ref",
    )
    provenance = value.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("artifact ref provenance must be an object")
    return ArtifactRef(
        artifact_id=_mapping_text(value, "artifact_id", "artifact ref"),
        run_id=_mapping_text(value, "run_id", "artifact ref"),
        conversation_id=_mapping_text(value, "conversation_id", "artifact ref"),
        call_id=_mapping_text(value, "call_id", "artifact ref"),
        capability_id=_mapping_text(value, "capability_id", "artifact ref"),
        filename=_mapping_text(value, "filename", "artifact ref"),
        media_type=_mapping_text(value, "media_type", "artifact ref"),
        byte_size=_mapping_int(value, "byte_size", "artifact ref"),
        sha256=_mapping_text(value, "sha256", "artifact ref"),
        sensitivity=Sensitivity(value.get("sensitivity")),
        provenance=artifact_provenance_from_mapping(provenance),
        created_at=_datetime_value(value.get("created_at"), "artifact created_at"),
    )


def artifact_destination_to_mapping(value: ArtifactDestination) -> dict[str, object]:
    return {
        "destination_id": value.destination_id,
        "display_name": value.display_name,
        "kind": value.kind.value,
        "authorization": value.authorization.value,
        "availability": value.availability.value,
        "is_default": value.is_default,
    }


def artifact_delivery_receipt_to_mapping(
    value: ArtifactDeliveryReceipt,
) -> dict[str, object]:
    return {
        "artifact_id": value.artifact_id,
        "destination_id": value.destination_id,
        "filename": value.filename,
        "saved_path": value.saved_path,
        "byte_size": value.byte_size,
        "sha256": value.sha256,
        "renamed_for_collision": value.renamed_for_collision,
        "delivered_at": _datetime_text(value.delivered_at),
    }


def artifact_delivery_receipt_from_mapping(
    value: Mapping[str, object],
) -> ArtifactDeliveryReceipt:
    _exact_keys(
        value,
        frozenset(
            {
                "artifact_id",
                "destination_id",
                "filename",
                "saved_path",
                "byte_size",
                "sha256",
                "renamed_for_collision",
                "delivered_at",
            }
        ),
        "artifact delivery receipt",
    )
    renamed = value.get("renamed_for_collision")
    if not isinstance(renamed, bool):
        raise ValueError("artifact delivery renamed_for_collision must be a boolean")
    return ArtifactDeliveryReceipt(
        artifact_id=_mapping_text(value, "artifact_id", "artifact delivery"),
        destination_id=_mapping_text(value, "destination_id", "artifact delivery"),
        filename=_mapping_text(value, "filename", "artifact delivery"),
        saved_path=_mapping_text(value, "saved_path", "artifact delivery"),
        byte_size=_mapping_int(value, "byte_size", "artifact delivery"),
        sha256=_mapping_text(value, "sha256", "artifact delivery"),
        renamed_for_collision=renamed,
        delivered_at=_datetime_value(
            value.get("delivered_at"), "delivery delivered_at"
        ),
    )


__all__ = [
    "ArtifactAuthorship",
    "ArtifactDeliveryReceipt",
    "ArtifactDestination",
    "ArtifactDestinationKind",
    "ArtifactDraft",
    "ArtifactError",
    "ArtifactPayload",
    "ArtifactProvenance",
    "ArtifactRef",
    "ArtifactResourceBinding",
    "DestinationAuthorization",
    "DestinationAvailability",
]
