"""Fixed document and exact-CSV rendering functions; deliberately no registry."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Sequence
from datetime import date, datetime, time
from decimal import Decimal
import math
import re
import time as time_module
from typing import cast
from uuid import UUID

from ..catalog.models import Sensitivity
from .models import (
    MAX_ARTIFACT_BYTES,
    MAX_DOCUMENT_BYTES,
    MAX_DOCUMENT_CHARACTERS,
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
    ArtifactProvenance,
    canonical_artifact_filename,
)

DOCUMENT_ALLOWED_EXTENSIONS = (
    ("text/markdown", (".md",)),
    ("text/plain", (".txt",)),
)
CSV_ALLOWED_EXTENSIONS = (("text/csv", (".csv",)),)
MAX_CSV_ROWS = 100_000
MAX_CSV_COLUMNS = 256
MAX_CSV_BYTES = MAX_ARTIFACT_BYTES
MAX_CSV_SECONDS = 60.0

_FORMULA_DANGEROUS = re.compile(r"^'*[ \t\r\n]*[=+\-@]")


def render_model_document(
    *,
    content: str,
    format: str,
    filename: str | None,
    evidence_call_ids: tuple[str, ...],
) -> ArtifactDraft:
    """Render one bounded UTF-8 narrative with normalized LF newlines."""

    if not isinstance(content, str):
        raise ArtifactError(
            "artifact_invalid_format",
            "Document content must be Unicode text.",
            {"media_type": "text/plain", "allowed_extensions": (".txt", ".md")},
        )
    if not content:
        raise ArtifactError(
            "artifact_invalid_format",
            "Document content must be non-empty.",
            {"media_type": "text/plain", "allowed_extensions": (".txt", ".md")},
        )
    if len(content) > MAX_DOCUMENT_CHARACTERS:
        raise ArtifactError(
            "artifact_quota_exceeded",
            "The model-authored document exceeds its character limit.",
            {
                "scope": "call",
                "limit_kind": "characters",
                "limit": MAX_DOCUMENT_CHARACTERS,
                "attempted": len(content),
            },
        )
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")
    try:
        encoded = normalized.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ArtifactError(
            "artifact_invalid_format",
            "Document content is not valid UTF-8 text.",
            {"media_type": "text/plain", "allowed_extensions": (".txt", ".md")},
        ) from error
    if len(encoded) > MAX_DOCUMENT_BYTES:
        raise ArtifactError(
            "artifact_quota_exceeded",
            "The model-authored document exceeds its byte limit.",
            {
                "scope": "call",
                "limit_kind": "bytes",
                "limit": MAX_DOCUMENT_BYTES,
                "attempted": len(encoded),
            },
        )
    if format == "markdown":
        media_type = "text/markdown"
        requested_filename = filename or "analysis.md"
    elif format == "txt":
        media_type = "text/plain"
        requested_filename = filename or "analysis.txt"
    else:
        raise ArtifactError(
            "artifact_invalid_format",
            "The requested document format is not supported.",
            {
                "media_type": str(format),
                "allowed_extensions": (".md", ".txt"),
            },
        )
    safe_filename = canonical_artifact_filename(
        requested_filename,
        media_type,
        DOCUMENT_ALLOWED_EXTENSIONS,
    )
    return ArtifactDraft(
        content=encoded,
        suggested_filename=safe_filename,
        media_type=media_type,
        sensitivity=Sensitivity.INTERNAL,
        provenance=ArtifactProvenance(
            authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
            evidence_call_ids=evidence_call_ids,
        ),
    )


class ExactCsvRenderer:
    """Incrementally render one complete typed result into the frozen CSV dialect."""

    def __init__(
        self,
        columns: Sequence[str],
        *,
        max_rows: int = MAX_CSV_ROWS,
        max_columns: int = MAX_CSV_COLUMNS,
        max_bytes: int = MAX_CSV_BYTES,
        max_seconds: float = MAX_CSV_SECONDS,
        clock: Callable[[], float] = time_module.monotonic,
    ) -> None:
        for value, name in (
            (max_rows, "max_rows"),
            (max_columns, "max_columns"),
            (max_bytes, "max_bytes"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(max_seconds, (int, float))
            or isinstance(max_seconds, bool)
            or not math.isfinite(float(max_seconds))
            or float(max_seconds) <= 0
        ):
            raise ValueError("max_seconds must be finite and positive")
        if not callable(clock):
            raise TypeError("clock must be callable")

        self._clock = clock
        self._deadline = clock() + float(max_seconds)
        self._max_rows = max_rows
        self._max_columns = max_columns
        self._max_bytes = max_bytes
        self._columns = _validated_csv_columns(tuple(columns), max_columns)
        self._rows = 0
        self._content = bytearray()
        self._check_time()
        self._append_record(
            tuple(
                _quoted_csv_text(column, protect_backslash=False)
                for column in self._columns
            ),
            reason="byte_limit",
        )

    @property
    def columns(self) -> tuple[str, ...]:
        return self._columns

    @property
    def row_count(self) -> int:
        return self._rows

    @property
    def byte_count(self) -> int:
        return len(self._content)

    def append(self, row: Sequence[object]) -> None:
        """Append one complete source row or fail without returning partial bytes."""

        self._check_time()
        if self._rows >= self._max_rows:
            self.incomplete("row_limit")
        if isinstance(row, (str, bytes, bytearray, memoryview)) or not isinstance(
            row, Sequence
        ):
            raise ArtifactError(
                "artifact_unsupported_value",
                "The source returned a non-tabular CSV row.",
                {
                    "row_index": self._rows,
                    "column_index": 0,
                    "column_name": self._columns[0],
                    "runtime_type": _safe_runtime_type(row),
                },
            )
        values = tuple(row)
        if len(values) != len(self._columns):
            raise ArtifactError(
                "artifact_unsupported_value",
                "The source returned a CSV row with an invalid shape.",
                {
                    "row_index": self._rows,
                    "column_index": 0,
                    "column_name": self._columns[0],
                    "runtime_type": _safe_runtime_type(row),
                },
            )
        fields = tuple(
            _csv_scalar(
                value,
                row_index=self._rows,
                column_index=index,
                column_name=self._columns[index],
            )
            for index, value in enumerate(values)
        )
        self._check_time()
        self._append_record(fields, reason="byte_limit")
        self._rows += 1

    def incomplete(self, reason: str) -> None:
        """Report that source completion could not be proven."""

        raise ArtifactError(
            "artifact_incomplete_export",
            "The exact CSV export could not be completed within its fixed bounds.",
            {
                "reason": reason,
                "completed_rows": self._rows,
                "completed_columns": len(self._columns),
                "completed_bytes": len(self._content),
            },
        )

    def finish(self) -> bytes:
        """Return bytes only after the caller has exhausted the source cursor."""

        self._check_time()
        return bytes(self._content)

    def _append_record(self, fields: tuple[str, ...], *, reason: str) -> None:
        try:
            encoded = (",".join(fields) + "\r\n").encode("utf-8")
        except UnicodeEncodeError as error:
            raise ArtifactError(
                "artifact_unsupported_value",
                "The source returned text that cannot be encoded losslessly as UTF-8.",
                {
                    "row_index": self._rows,
                    "column_index": 0,
                    "column_name": self._columns[0],
                    "runtime_type": "str",
                },
            ) from error
        attempted = len(self._content) + len(encoded)
        if attempted > self._max_bytes:
            self.incomplete(reason)
        self._content.extend(encoded)

    def _check_time(self) -> None:
        if self._clock() >= self._deadline:
            self.incomplete("time_limit")


def render_exact_csv(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    max_rows: int = MAX_CSV_ROWS,
    max_columns: int = MAX_CSV_COLUMNS,
    max_bytes: int = MAX_CSV_BYTES,
    max_seconds: float = MAX_CSV_SECONDS,
    clock: Callable[[], float] = time_module.monotonic,
) -> bytes:
    """Render an exhausted finite iterable as one deterministic exact CSV file."""

    renderer = ExactCsvRenderer(
        columns,
        max_rows=max_rows,
        max_columns=max_columns,
        max_bytes=max_bytes,
        max_seconds=max_seconds,
        clock=clock,
    )
    for row in rows:
        renderer.append(row)
    return renderer.finish()


def _validated_csv_columns(
    columns: tuple[str, ...],
    max_columns: int,
) -> tuple[str, ...]:
    if not columns:
        raise ArtifactError(
            "artifact_unsupported_value",
            "The exact CSV result must contain at least one named column.",
            {"column_index": 0, "reason": "missing_columns", "runtime_type": "str"},
        )
    if len(columns) > max_columns:
        raise ArtifactError(
            "artifact_quota_exceeded",
            "The exact CSV result exceeds its column limit.",
            {
                "scope": "artifact",
                "limit_kind": "columns",
                "limit": max_columns,
                "attempted": len(columns),
            },
        )
    seen: set[str] = set()
    for index, column in enumerate(columns):
        reason: str | None = None
        if not isinstance(column, str):
            reason = "not_text"
        elif not column.strip():
            reason = "empty"
        elif len(column) > 256:
            reason = "too_long"
        elif column in seen:
            reason = "duplicate"
        else:
            try:
                column.encode("utf-8")
            except UnicodeEncodeError:
                reason = "invalid_unicode"
        if reason is not None:
            raise ArtifactError(
                "artifact_unsupported_value",
                "The exact CSV result contains an unsupported column name.",
                {
                    "column_index": index,
                    "reason": reason,
                    "runtime_type": _safe_runtime_type(column),
                },
            )
        seen.add(column)
    return columns


def _csv_scalar(
    value: object,
    *,
    row_index: int,
    column_index: int,
    column_name: str,
) -> str:
    if value is None:
        return r"\N"
    if type(value) is str:
        try:
            value.encode("utf-8")
        except UnicodeEncodeError:
            return _unsupported_csv_value(value, row_index, column_index, column_name)
        return _quoted_csv_text(value, protect_backslash=True)
    if type(value) is bool:
        return "TRUE" if value else "FALSE"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        if math.isfinite(value):
            return repr(value)
        return _unsupported_csv_value(value, row_index, column_index, column_name)
    if type(value) is Decimal:
        if value.is_finite():
            return format(value, "f")
        return _unsupported_csv_value(value, row_index, column_index, column_name)
    if type(value) is datetime:
        return _quoted_csv_text(
            value.isoformat(timespec="microseconds"), protect_backslash=False
        )
    if type(value) is date:
        return value.isoformat()
    if type(value) is time:
        return _quoted_csv_text(
            value.isoformat(timespec="microseconds"), protect_backslash=False
        )
    if type(value) in {bytes, bytearray, memoryview}:
        binary = bytes(cast(bytes | bytearray | memoryview, value))
        return r"\B" + base64.b64encode(binary).decode("ascii")
    if type(value) is UUID:
        return _quoted_csv_text(str(value).lower(), protect_backslash=False)
    return _unsupported_csv_value(value, row_index, column_index, column_name)


def _quoted_csv_text(value: str, *, protect_backslash: bool) -> str:
    escaped = ("\\" + value) if protect_backslash and value.startswith("\\") else value
    if _FORMULA_DANGEROUS.match(escaped):
        escaped = "'" + escaped
    return '"' + escaped.replace('"', '""') + '"'


def _unsupported_csv_value(
    value: object,
    row_index: int,
    column_index: int,
    column_name: str,
) -> str:
    raise ArtifactError(
        "artifact_unsupported_value",
        "The source returned a value without a lossless exact CSV representation.",
        {
            "row_index": row_index,
            "column_index": column_index,
            "column_name": column_name,
            "runtime_type": _safe_runtime_type(value),
        },
    )


def _safe_runtime_type(value: object) -> str:
    name = type(value).__name__
    return name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}", name) else "unknown"


__all__ = [
    "CSV_ALLOWED_EXTENSIONS",
    "DOCUMENT_ALLOWED_EXTENSIONS",
    "ExactCsvRenderer",
    "MAX_CSV_BYTES",
    "MAX_CSV_COLUMNS",
    "MAX_CSV_ROWS",
    "MAX_CSV_SECONDS",
    "render_exact_csv",
    "render_model_document",
]
