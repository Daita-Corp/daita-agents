"""Deterministic tabular comparison over accepted evidence datasets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol, cast

from ..._json import FrozenJsonObject, canonical_json, thaw_json
from ...capabilities import (
    AccessMode,
    Capability,
    EvidenceArtifact,
    EvidenceCandidate,
    ExtensionDeclarations,
    ExecutionRequest,
    Executor,
    RiskLevel,
    ToolView,
)

TABULAR_COMPARE_CAPABILITY_ID = "data.tabular.compare"
TABULAR_COMPARE_EVIDENCE_KIND = "data.tabular.comparison"
TABULAR_COMPARE_EXECUTOR_ID = "data.tabular.compare.executor"
TABULAR_COMPARE_TOOL_NAME = "data_compare_tabular"
TABULAR_COMPARISON_MEDIA_TYPE = "application/vnd.daita.tabular-comparison+json"

_MAX_COLUMNS = 512
_MAX_COMPARE_COLUMNS = 64
_MAX_DATASET_ROWS = 10_000
_MAX_RESOURCES = 1_000


def _required_text(value: str, field_name: str, *, maximum: int = 2_048) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")


def _sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 71
        or not value.startswith("sha256:")
        or any(character not in "0123456789abcdef" for character in value[7:])
    ):
        raise ValueError(f"{field_name} must use sha256")


def _texts(
    values: tuple[str, ...],
    field_name: str,
    *,
    maximum_items: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{field_name} cannot be empty")
    if len(result) > maximum_items:
        raise ValueError(f"{field_name} exceed {maximum_items} items")
    for value in result:
        _required_text(value, field_name, maximum=256)
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} cannot contain duplicates")
    return result


def _resource_revisions(
    values: tuple[tuple[str, str], ...],
) -> tuple[tuple[str, str], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("resource_revisions must be a sequence of pairs")
    result = cast(
        tuple[tuple[str, str], ...],
        tuple(sorted(tuple(value) for value in values)),
    )
    if not result:
        raise ValueError("resource_revisions cannot be empty")
    if len(result) > _MAX_RESOURCES:
        raise ValueError(f"resource_revisions exceed {_MAX_RESOURCES} items")
    for item in result:
        if len(item) != 2:
            raise ValueError("resource_revisions must contain pairs")
        _required_text(item[0], "resource_id", maximum=512)
        _sha256(item[1], "resource_revision")
    if len({item[0] for item in result}) != len(result):
        raise ValueError("resource_revisions cannot repeat a resource")
    return result


@dataclass(frozen=True, slots=True)
class TabularEvidenceDataset:
    """Rows loaded from one accepted, current-operation evidence record."""

    operation_id: str
    evidence_id: str
    evidence_kind: str
    source_id: str
    source_revision: str
    resource_revisions: tuple[tuple[str, str], ...]
    columns: tuple[str, ...]
    rows: tuple[Mapping[str, object], ...]
    complete: bool
    truncation_reasons: tuple[str, ...]
    row_limit: int
    byte_limit: int
    sensitivity_class: str = "internal"
    retention_class: str = "operation"

    def __post_init__(self) -> None:
        for value, name, maximum in (
            (self.operation_id, "operation_id", 512),
            (self.evidence_id, "evidence_id", 512),
            (self.evidence_kind, "evidence_kind", 256),
            (self.source_id, "source_id", 512),
            (self.source_revision, "source_revision", 1_024),
            (self.sensitivity_class, "sensitivity_class", 128),
            (self.retention_class, "retention_class", 128),
        ):
            _required_text(value, name, maximum=maximum)
        resources = _resource_revisions(self.resource_revisions)
        columns = _texts(
            self.columns,
            "columns",
            maximum_items=_MAX_COLUMNS,
            allow_empty=True,
        )
        if isinstance(self.rows, (str, bytes)):
            raise TypeError("rows must be a sequence of mappings")
        raw_rows = tuple(self.rows)
        if len(raw_rows) > _MAX_DATASET_ROWS:
            raise ValueError(f"rows exceed {_MAX_DATASET_ROWS} items")
        if any(not isinstance(row, Mapping) for row in raw_rows):
            raise TypeError("rows must contain mappings")
        rows = tuple(FrozenJsonObject.from_mapping(row) for row in raw_rows)
        if not isinstance(self.complete, bool):
            raise TypeError("complete must be a boolean")
        reasons = _texts(
            self.truncation_reasons,
            "truncation_reasons",
            maximum_items=16,
            allow_empty=True,
        )
        if self.complete == bool(reasons):
            raise ValueError("complete must agree with truncation_reasons")
        for limit_value, limit_name, minimum in (
            (self.row_limit, "row_limit", 1),
            (self.byte_limit, "byte_limit", 2),
        ):
            if (
                not isinstance(limit_value, int)
                or isinstance(limit_value, bool)
                or limit_value < minimum
            ):
                raise ValueError(
                    f"{limit_name} must be an integer of at least {minimum}"
                )
        if len(rows) > self.row_limit:
            raise ValueError("rows exceed row_limit")
        object.__setattr__(self, "resource_revisions", resources)
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "rows", rows)
        object.__setattr__(self, "truncation_reasons", reasons)

    def provenance_payload(self) -> dict[str, object]:
        return {
            "byte_limit": self.byte_limit,
            "complete": self.complete,
            "evidence_id": self.evidence_id,
            "evidence_kind": self.evidence_kind,
            "resource_revisions": tuple(
                {"resource_id": resource_id, "revision": revision}
                for resource_id, revision in self.resource_revisions
            ),
            "row_count": len(self.rows),
            "row_limit": self.row_limit,
            "source_id": self.source_id,
            "source_revision": self.source_revision,
            "truncation_reasons": self.truncation_reasons,
        }


class AcceptedEvidenceDatasetReader(Protocol):
    """Resolve comparison inputs from authoritative accepted evidence only."""

    async def load_dataset(
        self,
        *,
        operation_id: str,
        evidence_id: str,
    ) -> TabularEvidenceDataset: ...


@dataclass(frozen=True, slots=True)
class TabularComparisonResult:
    payload: FrozenJsonObject
    artifact: EvidenceArtifact
    discrepancies: tuple[FrozenJsonObject, ...]
    total_discrepancies: int
    complete: bool
    truncation_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.payload, FrozenJsonObject):
            raise TypeError("payload must be a FrozenJsonObject")
        if not isinstance(self.artifact, EvidenceArtifact):
            raise TypeError("artifact must be an EvidenceArtifact")
        discrepancies = tuple(self.discrepancies)
        if any(not isinstance(item, FrozenJsonObject) for item in discrepancies):
            raise TypeError("discrepancies must contain FrozenJsonObject records")
        if (
            not isinstance(self.total_discrepancies, int)
            or isinstance(self.total_discrepancies, bool)
            or self.total_discrepancies < len(discrepancies)
        ):
            raise ValueError("total_discrepancies is invalid")
        if not isinstance(self.complete, bool):
            raise TypeError("complete must be a boolean")
        reasons = tuple(self.truncation_reasons)
        if self.complete == bool(reasons):
            raise ValueError("complete must agree with truncation_reasons")
        object.__setattr__(self, "discrepancies", discrepancies)
        object.__setattr__(self, "truncation_reasons", reasons)


@dataclass(frozen=True, slots=True)
class TabularComparisonDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


class TabularComparisonExecutor:
    executor_id = TABULAR_COMPARE_EXECUTOR_ID

    def __init__(
        self,
        reader: AcceptedEvidenceDatasetReader,
        *,
        max_discrepancies: int = 1_000,
        max_inline_discrepancies: int = 20,
        max_artifact_bytes: int = 4 * 1_024 * 1_024,
    ) -> None:
        if not callable(getattr(reader, "load_dataset", None)):
            raise TypeError("reader must provide load_dataset")
        for value, name, minimum in (
            (max_discrepancies, "max_discrepancies", 1),
            (max_inline_discrepancies, "max_inline_discrepancies", 0),
            (max_artifact_bytes, "max_artifact_bytes", 4_096),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if max_inline_discrepancies > max_discrepancies:
            raise ValueError("inline discrepancy bound cannot exceed artifact bound")
        self._reader = reader
        self._max_discrepancies = max_discrepancies
        self._max_inline_discrepancies = max_inline_discrepancies
        self._max_artifact_bytes = max_artifact_bytes

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        left_id = request.arguments["left_evidence_id"]
        right_id = request.arguments["right_evidence_id"]
        key_columns = request.arguments["key_columns"]
        compare_columns = request.arguments["compare_columns"]
        assert isinstance(left_id, str)
        assert isinstance(right_id, str)
        assert isinstance(key_columns, tuple)
        assert isinstance(compare_columns, tuple)
        _required_text(left_id, "left_evidence_id", maximum=512)
        _required_text(right_id, "right_evidence_id", maximum=512)
        if left_id == right_id:
            raise ValueError("comparison evidence inputs must be distinct")
        keys = _texts(
            key_columns,
            "key_columns",
            maximum_items=_MAX_COMPARE_COLUMNS,
        )
        compared = _texts(
            compare_columns,
            "compare_columns",
            maximum_items=_MAX_COMPARE_COLUMNS,
        )
        if set(keys) & set(compared):
            raise ValueError("key_columns and compare_columns cannot overlap")
        left = await self._reader.load_dataset(
            operation_id=request.operation_id,
            evidence_id=left_id,
        )
        right = await self._reader.load_dataset(
            operation_id=request.operation_id,
            evidence_id=right_id,
        )
        for dataset, evidence_id, side in (
            (left, left_id, "left"),
            (right, right_id, "right"),
        ):
            if not isinstance(dataset, TabularEvidenceDataset):
                raise TypeError(f"{side} reader result must be TabularEvidenceDataset")
            if (
                dataset.operation_id != request.operation_id
                or dataset.evidence_id != evidence_id
            ):
                raise ValueError(f"{side} dataset escaped requested evidence scope")
        result = compare_tabular_datasets(
            left,
            right,
            key_columns=keys,
            compare_columns=compared,
            max_discrepancies=self._max_discrepancies,
            max_inline_discrepancies=self._max_inline_discrepancies,
            max_artifact_bytes=self._max_artifact_bytes,
        )
        return EvidenceCandidate(
            kind=TABULAR_COMPARE_EVIDENCE_KIND,
            schema_version=1,
            payload=result.payload,
            artifact=result.artifact,
        )


def compare_tabular_datasets(
    left: TabularEvidenceDataset,
    right: TabularEvidenceDataset,
    *,
    key_columns: tuple[str, ...],
    compare_columns: tuple[str, ...],
    max_discrepancies: int = 1_000,
    max_inline_discrepancies: int = 20,
    max_artifact_bytes: int = 4 * 1_024 * 1_024,
) -> TabularComparisonResult:
    if not isinstance(left, TabularEvidenceDataset) or not isinstance(
        right,
        TabularEvidenceDataset,
    ):
        raise TypeError("comparison inputs must be TabularEvidenceDataset records")
    keys = _texts(
        key_columns,
        "key_columns",
        maximum_items=_MAX_COMPARE_COLUMNS,
    )
    compared = _texts(
        compare_columns,
        "compare_columns",
        maximum_items=_MAX_COMPARE_COLUMNS,
    )
    if set(keys) & set(compared):
        raise ValueError("key_columns and compare_columns cannot overlap")
    for dataset, side in ((left, "left"), (right, "right")):
        missing = tuple(
            column for column in (*keys, *compared) if column not in dataset.columns
        )
        if missing:
            raise ValueError(
                f"{side} dataset lacks declared columns: {', '.join(missing)}"
            )
    for value, name, minimum in (
        (max_discrepancies, "max_discrepancies", 1),
        (max_inline_discrepancies, "max_inline_discrepancies", 0),
        (max_artifact_bytes, "max_artifact_bytes", 4_096),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer of at least {minimum}")
    if max_inline_discrepancies > max_discrepancies:
        raise ValueError("inline discrepancy bound cannot exceed artifact bound")

    stored: list[FrozenJsonObject] = []
    total_discrepancies = 0

    def record(payload: Mapping[str, object]) -> None:
        nonlocal total_discrepancies
        total_discrepancies += 1
        if len(stored) < max_discrepancies:
            stored.append(FrozenJsonObject.from_mapping(payload))

    left_index, left_invalid = _index_rows(left, "left", keys, record)
    right_index, right_invalid = _index_rows(right, "right", keys, record)
    left_duplicates = _record_duplicates(left_index, "left", record)
    right_duplicates = _record_duplicates(right_index, "right", record)

    left_keys = set(left_index)
    right_keys = set(right_index)
    left_only = 0
    for fingerprint in sorted(left_keys - right_keys):
        rows = left_index[fingerprint]
        if len(rows) != 1:
            continue
        left_only += 1
        record(
            {
                "kind": "left_only",
                "key": rows[0][2],
                "left_row_index": rows[0][0],
            }
        )
    right_only = 0
    for fingerprint in sorted(right_keys - left_keys):
        rows = right_index[fingerprint]
        if len(rows) != 1:
            continue
        right_only += 1
        record(
            {
                "kind": "right_only",
                "key": rows[0][2],
                "right_row_index": rows[0][0],
            }
        )

    matched_keys = 0
    equal_rows = 0
    different_rows = 0
    value_mismatches = 0
    for fingerprint in sorted(left_keys & right_keys):
        left_group = left_index[fingerprint]
        right_group = right_index[fingerprint]
        if len(left_group) != 1 or len(right_group) != 1:
            continue
        matched_keys += 1
        left_index_value, left_row, key_payload = left_group[0]
        right_index_value, right_row, _ = right_group[0]
        row_different = False
        for column in compared:
            left_present = column in left_row
            right_present = column in right_row
            left_value = left_row.get(column)
            right_value = right_row.get(column)
            if (
                left_present
                and right_present
                and _strict_equal(left_value, right_value)
            ):
                continue
            row_different = True
            value_mismatches += 1
            kind = (
                "missing_value"
                if not left_present or not right_present
                else (
                    "type_mismatch"
                    if _json_type(left_value) != _json_type(right_value)
                    else "value_mismatch"
                )
            )
            record(
                {
                    "column": column,
                    "key": key_payload,
                    "kind": kind,
                    "left_present": left_present,
                    "left_row_index": left_index_value,
                    "left_type": (
                        "missing" if not left_present else _json_type(left_value)
                    ),
                    "left_value": left_value,
                    "right_present": right_present,
                    "right_row_index": right_index_value,
                    "right_type": (
                        "missing" if not right_present else _json_type(right_value)
                    ),
                    "right_value": right_value,
                }
            )
        if row_different:
            different_rows += 1
        else:
            equal_rows += 1

    counts = {
        "different_rows": different_rows,
        "equal_rows": equal_rows,
        "left_duplicate_keys": left_duplicates,
        "left_invalid_keys": left_invalid,
        "left_only": left_only,
        "left_rows": len(left.rows),
        "matched_keys": matched_keys,
        "right_duplicate_keys": right_duplicates,
        "right_invalid_keys": right_invalid,
        "right_only": right_only,
        "right_rows": len(right.rows),
        "value_mismatches": value_mismatches,
    }
    reasons = [
        *(f"left:{reason}" for reason in left.truncation_reasons),
        *(f"right:{reason}" for reason in right.truncation_reasons),
    ]
    if total_discrepancies > len(stored):
        reasons.append("discrepancy_limit")
    artifact_discrepancies = tuple(stored)
    artifact_bytes = _comparison_artifact_bytes(
        left,
        right,
        keys,
        compared,
        counts,
        artifact_discrepancies,
        total_discrepancies,
        tuple(dict.fromkeys(reasons)),
    )
    if len(artifact_bytes) > max_artifact_bytes:
        if "artifact_byte_limit" not in reasons:
            reasons.append("artifact_byte_limit")
        artifact_discrepancies = _fit_artifact_discrepancies(
            left,
            right,
            keys,
            compared,
            counts,
            artifact_discrepancies,
            total_discrepancies,
            tuple(dict.fromkeys(reasons)),
            max_artifact_bytes,
        )
        artifact_bytes = _comparison_artifact_bytes(
            left,
            right,
            keys,
            compared,
            counts,
            artifact_discrepancies,
            total_discrepancies,
            tuple(dict.fromkeys(reasons)),
        )
        if len(artifact_bytes) > max_artifact_bytes:
            raise ValueError("comparison provenance exceeds artifact byte limit")

    reason_tuple = tuple(dict.fromkeys(reasons))
    complete = not reason_tuple
    digest = "sha256:" + sha256(artifact_bytes).hexdigest()
    artifact = EvidenceArtifact(
        content=artifact_bytes,
        media_type=TABULAR_COMPARISON_MEDIA_TYPE,
        sensitivity_class=_combined_sensitivity(left, right),
        retention_class=(
            left.retention_class
            if left.retention_class == right.retention_class
            else "operation"
        ),
    )
    payload = FrozenJsonObject.from_mapping(
        {
            "artifact_digest": digest,
            "artifact_media_type": TABULAR_COMPARISON_MEDIA_TYPE,
            "compare_columns": compared,
            "complete": complete,
            "counts": counts,
            "discrepancy_sample": tuple(
                thaw_json(item)
                for item in artifact_discrepancies[:max_inline_discrepancies]
            ),
            "key_columns": keys,
            "left": left.provenance_payload(),
            "right": right.provenance_payload(),
            "stored_discrepancies": len(artifact_discrepancies),
            "total_discrepancies": total_discrepancies,
            "truncated": not complete,
            "truncation_reasons": reason_tuple,
            "trust_classification": "untrusted_external_data",
        }
    )
    return TabularComparisonResult(
        payload=payload,
        artifact=artifact,
        discrepancies=artifact_discrepancies,
        total_discrepancies=total_discrepancies,
        complete=complete,
        truncation_reasons=reason_tuple,
    )


def _index_rows(
    dataset: TabularEvidenceDataset,
    side: str,
    keys: tuple[str, ...],
    record: object,
) -> tuple[
    dict[str, list[tuple[int, FrozenJsonObject, FrozenJsonObject]]],
    int,
]:
    recorder = record
    assert callable(recorder)
    indexed: dict[str, list[tuple[int, FrozenJsonObject, FrozenJsonObject]]] = {}
    invalid = 0
    for row_index, raw_row in enumerate(dataset.rows):
        # __post_init__ freezes every accepted row before a dataset is observable.
        row = cast(FrozenJsonObject, raw_row)
        missing = tuple(column for column in keys if column not in row)
        nulls = tuple(
            column for column in keys if column in row and row[column] is None
        )
        if missing or nulls:
            invalid += 1
            recorder(
                {
                    "kind": "invalid_key",
                    "missing_columns": missing,
                    "null_columns": nulls,
                    "row_index": row_index,
                    "side": side,
                }
            )
            continue
        values = tuple(row[column] for column in keys)
        fingerprint = canonical_json(values)
        key_payload = FrozenJsonObject.from_mapping(
            {column: row[column] for column in keys}
        )
        indexed.setdefault(fingerprint, []).append((row_index, row, key_payload))
    return indexed, invalid


def _record_duplicates(
    indexed: Mapping[
        str,
        list[tuple[int, FrozenJsonObject, FrozenJsonObject]],
    ],
    side: str,
    record: object,
) -> int:
    recorder = record
    assert callable(recorder)
    count = 0
    for fingerprint in sorted(indexed):
        rows = indexed[fingerprint]
        if len(rows) < 2:
            continue
        count += 1
        recorder(
            {
                "key": rows[0][2],
                "kind": "duplicate_key",
                "row_indexes": tuple(row[0] for row in rows),
                "side": side,
            }
        )
    return count


def _json_type(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, (tuple, list)):
        return "array"
    if isinstance(value, Mapping):
        return "object"
    raise TypeError(f"unsupported comparison JSON value: {type(value).__name__}")


def _strict_equal(left: object, right: object) -> bool:
    return _json_type(left) == _json_type(right) and canonical_json(
        left
    ) == canonical_json(right)


def _artifact_payload(
    left: TabularEvidenceDataset,
    right: TabularEvidenceDataset,
    keys: tuple[str, ...],
    compared: tuple[str, ...],
    counts: Mapping[str, object],
    discrepancies: tuple[FrozenJsonObject, ...],
    total_discrepancies: int,
    reasons: tuple[str, ...],
) -> dict[str, object]:
    return {
        "compare_columns": compared,
        "complete": not reasons,
        "counts": counts,
        "discrepancies": tuple(thaw_json(item) for item in discrepancies),
        "key_columns": keys,
        "left": left.provenance_payload(),
        "right": right.provenance_payload(),
        "schema_version": 1,
        "stored_discrepancies": len(discrepancies),
        "total_discrepancies": total_discrepancies,
        "truncated": bool(reasons),
        "truncation_reasons": reasons,
        "trust_classification": "untrusted_external_data",
    }


def _comparison_artifact_bytes(
    left: TabularEvidenceDataset,
    right: TabularEvidenceDataset,
    keys: tuple[str, ...],
    compared: tuple[str, ...],
    counts: Mapping[str, object],
    discrepancies: tuple[FrozenJsonObject, ...],
    total_discrepancies: int,
    reasons: tuple[str, ...],
) -> bytes:
    return canonical_json(
        _artifact_payload(
            left,
            right,
            keys,
            compared,
            counts,
            discrepancies,
            total_discrepancies,
            reasons,
        )
    ).encode("utf-8")


def _fit_artifact_discrepancies(
    left: TabularEvidenceDataset,
    right: TabularEvidenceDataset,
    keys: tuple[str, ...],
    compared: tuple[str, ...],
    counts: Mapping[str, object],
    discrepancies: tuple[FrozenJsonObject, ...],
    total_discrepancies: int,
    reasons: tuple[str, ...],
    maximum: int,
) -> tuple[FrozenJsonObject, ...]:
    low = 0
    high = len(discrepancies)
    while low < high:
        middle = (low + high + 1) // 2
        candidate = discrepancies[:middle]
        encoded = _comparison_artifact_bytes(
            left,
            right,
            keys,
            compared,
            counts,
            candidate,
            total_discrepancies,
            reasons,
        )
        if len(encoded) <= maximum:
            low = middle
        else:
            high = middle - 1
    return discrepancies[:low]


def _combined_sensitivity(
    left: TabularEvidenceDataset,
    right: TabularEvidenceDataset,
) -> str:
    ranks = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3,
        "unknown": 4,
    }
    return max(
        (left.sensitivity_class, right.sensitivity_class),
        key=lambda value: ranks.get(value, 4),
    )


def tabular_comparison_declarations(
    reader: AcceptedEvidenceDatasetReader,
) -> TabularComparisonDeclarations:
    executor = TabularComparisonExecutor(reader)
    extension = tabular_comparison_extension_declarations()
    return TabularComparisonDeclarations(
        capabilities=extension.capabilities,
        executors=(executor,),
        tool_views=extension.tool_views,
    )


def tabular_comparison_extension_declarations() -> ExtensionDeclarations:
    capability = Capability(
        id=TABULAR_COMPARE_CAPABILITY_ID,
        owner="data",
        description="Compare two accepted tabular evidence datasets deterministically.",
        input_schema={
            "type": "object",
            "properties": {
                "left_evidence_id": {"type": "string"},
                "right_evidence_id": {"type": "string"},
                "key_columns": {"type": "array"},
                "compare_columns": {"type": "array"},
            },
            "required": [
                "left_evidence_id",
                "right_evidence_id",
                "key_columns",
                "compare_columns",
            ],
            "additionalProperties": False,
        },
        output_evidence_kind=TABULAR_COMPARE_EVIDENCE_KIND,
        output_schema_version=1,
        output_schema=_comparison_output_schema(),
        executor_id=TABULAR_COMPARE_EXECUTOR_ID,
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )
    view = ToolView(
        name=TABULAR_COMPARE_TOOL_NAME,
        capability_id=capability.id,
        description=capability.description,
    )
    return ExtensionDeclarations(
        capabilities=(capability,),
        executor_ids=(TABULAR_COMPARE_EXECUTOR_ID,),
        tool_views=(view,),
    )


def _comparison_output_schema() -> dict[str, object]:
    properties = {
        "artifact_digest": {"type": "string"},
        "artifact_media_type": {"type": "string"},
        "compare_columns": {"type": "array"},
        "complete": {"type": "boolean"},
        "counts": {"type": "object"},
        "discrepancy_sample": {"type": "array"},
        "key_columns": {"type": "array"},
        "left": {"type": "object"},
        "right": {"type": "object"},
        "stored_discrepancies": {"type": "integer"},
        "total_discrepancies": {"type": "integer"},
        "truncated": {"type": "boolean"},
        "truncation_reasons": {"type": "array"},
        "trust_classification": {"type": "string"},
    }
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


__all__ = [
    "TABULAR_COMPARE_CAPABILITY_ID",
    "TABULAR_COMPARE_EVIDENCE_KIND",
    "TABULAR_COMPARE_EXECUTOR_ID",
    "TABULAR_COMPARE_TOOL_NAME",
    "TABULAR_COMPARISON_MEDIA_TYPE",
    "AcceptedEvidenceDatasetReader",
    "TabularComparisonDeclarations",
    "TabularComparisonExecutor",
    "TabularComparisonResult",
    "TabularEvidenceDataset",
    "compare_tabular_datasets",
    "tabular_comparison_declarations",
    "tabular_comparison_extension_declarations",
]
