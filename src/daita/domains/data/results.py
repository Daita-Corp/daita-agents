"""Deterministic bounded projection for untrusted tabular result rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from ..._json import FrozenJsonObject, canonical_json, thaw_json


@dataclass(frozen=True, slots=True)
class BoundedResultProjection:
    """An immutable complete-row projection with explicit limitation facts."""

    rows: tuple[FrozenJsonObject, ...]
    total_rows: int
    returned_rows: int
    utf8_bytes: int
    row_limit: int
    byte_limit: int
    truncated: bool
    truncation_reasons: tuple[str, ...]
    trust_classification: str = "untrusted_external_data"

    def __post_init__(self) -> None:
        if self.total_rows < 0 or self.returned_rows < 0:
            raise ValueError("result row counts must be non-negative")
        if self.returned_rows != len(self.rows):
            raise ValueError("returned_rows must match projected rows")
        if self.returned_rows > self.total_rows:
            raise ValueError("returned_rows cannot exceed total_rows")
        if self.row_limit < 1:
            raise ValueError("row_limit must be positive")
        if self.byte_limit < 2:
            raise ValueError("byte_limit must fit an empty JSON array")
        if self.utf8_bytes < 2 or self.utf8_bytes > self.byte_limit:
            raise ValueError("utf8_bytes must fit the configured byte limit")
        reasons = tuple(self.truncation_reasons)
        if len(reasons) != len(set(reasons)):
            raise ValueError("truncation reasons must be unique")
        if self.truncated != bool(reasons):
            raise ValueError("truncated must agree with truncation reasons")
        if self.truncated != (self.returned_rows < self.total_rows):
            raise ValueError("truncated must agree with projected row count")
        if self.trust_classification != "untrusted_external_data":
            raise ValueError("result rows must remain classified as untrusted data")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "truncation_reasons", reasons)

    def to_payload(self) -> FrozenJsonObject:
        """Return the strict JSON evidence payload without exposing mutable rows."""

        return FrozenJsonObject.from_mapping(
            {
                "rows": [thaw_json(row) for row in self.rows],
                "total_rows": self.total_rows,
                "returned_rows": self.returned_rows,
                "utf8_bytes": self.utf8_bytes,
                "row_limit": self.row_limit,
                "byte_limit": self.byte_limit,
                "truncated": self.truncated,
                "truncation_reasons": list(self.truncation_reasons),
                "trust_classification": self.trust_classification,
            }
        )


def project_result_rows(
    rows: Iterable[Mapping[str, object]],
    *,
    max_rows: int,
    max_bytes: int,
) -> BoundedResultProjection:
    """Project rows within row and canonical UTF-8 byte limits.

    The byte limit applies to the canonical JSON array. A row is included only
    when the entire row fits, so downstream consumers never receive fragments.
    """

    if not isinstance(max_rows, int) or isinstance(max_rows, bool) or max_rows < 1:
        raise ValueError("max_rows must be a positive integer")
    if not isinstance(max_bytes, int) or isinstance(max_bytes, bool) or max_bytes < 2:
        raise ValueError("max_bytes must be an integer of at least 2")

    frozen_rows: list[FrozenJsonObject] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"result row {index} must be a mapping")
        frozen_rows.append(FrozenJsonObject.from_mapping(row))

    total_rows = len(frozen_rows)
    row_candidates = frozen_rows[:max_rows]
    projected: list[FrozenJsonObject] = []
    byte_limited = False
    for row in row_candidates:
        candidate = [thaw_json(item) for item in (*projected, row)]
        candidate_bytes = len(canonical_json(candidate).encode("utf-8"))
        if candidate_bytes > max_bytes:
            byte_limited = True
            break
        projected.append(row)

    reasons: list[str] = []
    if total_rows > max_rows:
        reasons.append("row_limit")
    if byte_limited:
        reasons.append("byte_limit")
    encoded_bytes = len(
        canonical_json([thaw_json(item) for item in projected]).encode("utf-8")
    )
    return BoundedResultProjection(
        rows=tuple(projected),
        total_rows=total_rows,
        returned_rows=len(projected),
        utf8_bytes=encoded_bytes,
        row_limit=max_rows,
        byte_limit=max_bytes,
        truncated=bool(reasons),
        truncation_reasons=tuple(reasons),
    )
