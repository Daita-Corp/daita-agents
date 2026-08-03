"""Fixed document and exact-tabular renderers; deliberately no registry."""

from __future__ import annotations

import base64
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from decimal import Decimal
from hashlib import sha256
from importlib import import_module
from io import BytesIO
import math
import re
import stat
import time as time_module
from typing import Any, cast
from uuid import UUID
from xml.etree import ElementTree
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

from .._installation import PIPX_REPAIR_GUIDANCE
from .._json import canonical_json
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
XLSX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
XLSX_ALLOWED_EXTENSIONS = ((XLSX_MEDIA_TYPE, (".xlsx",)),)
MAX_CSV_ROWS = 100_000
MAX_CSV_COLUMNS = 256
MAX_CSV_BYTES = MAX_ARTIFACT_BYTES
MAX_CSV_SECONDS = 60.0
MAX_XLSX_ROWS = 100_000
MAX_XLSX_COLUMNS = 256
MAX_XLSX_BYTES = MAX_ARTIFACT_BYTES
MAX_XLSX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
MAX_XLSX_MEMBERS = 64
MAX_XLSX_SECONDS = 60.0
MAX_XLSX_TEXT_UTF16_UNITS = 32_767

_FORMULA_DANGEROUS = re.compile(r"^'*[ \t\r\n]*[=+\-@]")
_XLSX_MEMBER_ORDER = (
    "[Content_Types].xml",
    "_rels/.rels",
    "xl/_rels/workbook.xml.rels",
    "xl/worksheets/sheet1.xml",
    "xl/worksheets/sheet2.xml",
    "xl/workbook.xml",
    "xl/sharedStrings.xml",
    "xl/styles.xml",
    "xl/theme/theme1.xml",
    "docProps/core.xml",
    "docProps/app.xml",
)
_XLSX_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
_SPREADSHEET_NAMESPACE = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_RELATIONSHIP_NAMESPACE = "http://schemas.openxmlformats.org/package/2006/relationships"
_EXTENDED_PROPERTIES_NAMESPACE = (
    "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
)
_CORE_NAMESPACE = (
    "http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
)
_DC_NAMESPACE = "http://purl.org/dc/elements/1.1/"
_DCTERMS_NAMESPACE = "http://purl.org/dc/terms/"
_PROHIBITED_WORKSHEET_ELEMENTS = frozenset(
    {
        "dataValidations",
        "drawing",
        "extLst",
        "hyperlinks",
        "legacyDrawing",
        "legacyDrawingHF",
        "mergeCells",
        "oleObjects",
        "picture",
        "tableParts",
    }
)
_XLSX_RELATIONSHIPS = {
    "_rels/.rels": (
        (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument",
            "xl/workbook.xml",
        ),
        (
            "http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties",
            "docProps/core.xml",
        ),
        (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties",
            "docProps/app.xml",
        ),
    ),
    "xl/_rels/workbook.xml.rels": (
        (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
            "worksheets/sheet1.xml",
        ),
        (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet",
            "worksheets/sheet2.xml",
        ),
        (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme",
            "theme/theme1.xml",
        ),
        (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles",
            "styles.xml",
        ),
        (
            "http://schemas.openxmlformats.org/officeDocument/2006/relationships/sharedStrings",
            "sharedStrings.xml",
        ),
    ),
}


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


@dataclass(frozen=True, slots=True)
class ExactXlsxProvenance:
    """Runtime-owned safe facts used by the fixed XLSX provenance sheet."""

    source_id: str
    source_revision: str
    resource_revisions: tuple[tuple[str, str], ...]
    sql_fingerprint: str
    parameters_sha256: str
    sensitivity: Sensitivity
    created_at: datetime

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "source_id"),
            (self.source_revision, "source_revision"),
            (self.sql_fingerprint, "sql_fingerprint"),
            (self.parameters_sha256, "parameters_sha256"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"XLSX provenance {name} must be non-empty text")
        if not self.sql_fingerprint.startswith("sha256:"):
            raise ValueError("XLSX provenance sql_fingerprint must use sha256")
        if not self.parameters_sha256.startswith("sha256:"):
            raise ValueError("XLSX provenance parameters_sha256 must use sha256")
        revisions = tuple(sorted(tuple(item) for item in self.resource_revisions))
        if not revisions or len(revisions) > 64:
            raise ValueError("XLSX provenance resource revisions exceed their bound")
        if len({item[0] for item in revisions}) != len(revisions) or any(
            not resource_id or not revision.startswith("sha256:")
            for resource_id, revision in revisions
        ):
            raise ValueError("XLSX provenance resource revisions are invalid")
        if not isinstance(self.sensitivity, Sensitivity):
            raise TypeError("XLSX provenance sensitivity must be Sensitivity")
        if self.sensitivity is Sensitivity.UNKNOWN:
            raise ValueError("XLSX provenance sensitivity must be resolved")
        if (
            not isinstance(self.created_at, datetime)
            or self.created_at.tzinfo is None
            or self.created_at.utcoffset() is None
        ):
            raise ValueError("XLSX provenance created_at must be timezone-aware")
        object.__setattr__(self, "resource_revisions", revisions)
        object.__setattr__(
            self,
            "created_at",
            self.created_at.astimezone(timezone.utc).replace(microsecond=0),
        )


class ExactXlsxRenderer:
    """Render one literal-only deterministic workbook with fixed provenance."""

    def __init__(
        self,
        columns: Sequence[str],
        *,
        provenance: ExactXlsxProvenance,
        max_rows: int = MAX_XLSX_ROWS,
        max_columns: int = MAX_XLSX_COLUMNS,
        max_bytes: int = MAX_XLSX_BYTES,
        max_uncompressed_bytes: int = MAX_XLSX_UNCOMPRESSED_BYTES,
        max_members: int = MAX_XLSX_MEMBERS,
        max_seconds: float = MAX_XLSX_SECONDS,
        clock: Callable[[], float] = time_module.monotonic,
    ) -> None:
        for value, name in (
            (max_rows, "max_rows"),
            (max_columns, "max_columns"),
            (max_bytes, "max_bytes"),
            (max_uncompressed_bytes, "max_uncompressed_bytes"),
            (max_members, "max_members"),
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
        if not isinstance(provenance, ExactXlsxProvenance):
            raise TypeError("provenance must be ExactXlsxProvenance")

        self._clock = clock
        self._deadline = clock() + float(max_seconds)
        self._max_rows = max_rows
        self._max_bytes = max_bytes
        self._max_uncompressed_bytes = max_uncompressed_bytes
        self._max_members = max_members
        self._columns = _validated_tabular_columns(
            tuple(columns), max_columns, format_name="XLSX"
        )
        self._provenance = provenance
        self._rows = 0
        self._content = b""
        self._closed = False
        self._check_time()

        xlsxwriter = _load_xlsxwriter()
        self._buffer = BytesIO()
        self._workbook = xlsxwriter.Workbook(
            self._buffer,
            {
                "in_memory": True,
                "strings_to_formulas": False,
                "strings_to_urls": False,
            },
        )
        self._workbook.set_properties(
            {
                "title": "Daita exact tabular export",
                "subject": "Exact source data",
                "author": "Daita",
                "company": "Daita",
                "comments": "Deterministic literal-only workbook",
                "created": provenance.created_at,
                "modified": provenance.created_at,
            }
        )
        self._data = self._workbook.add_worksheet("Data")
        self._provenance_sheet = self._workbook.add_worksheet("Provenance")
        self._date_format = self._workbook.add_format({"num_format": "yyyy-mm-dd"})
        for column_index, column in enumerate(self._columns):
            _write_xlsx_text(
                self._data,
                0,
                column_index,
                column,
                row_index=-1,
                column_name=column,
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
        self._check_open()
        self._check_time()
        if self._rows >= self._max_rows:
            self.incomplete("row_limit")
        if isinstance(row, (str, bytes, bytearray, memoryview)) or not isinstance(
            row, Sequence
        ):
            raise _unsupported_xlsx_value(
                row,
                self._rows,
                0,
                self._columns[0],
                message="The source returned a non-tabular XLSX row.",
            )
        values = tuple(row)
        if len(values) != len(self._columns):
            raise _unsupported_xlsx_value(
                row,
                self._rows,
                0,
                self._columns[0],
                message="The source returned an XLSX row with an invalid shape.",
            )
        for column_index, value in enumerate(values):
            _write_xlsx_scalar(
                self._data,
                self._rows + 1,
                column_index,
                value,
                date_format=self._date_format,
                row_index=self._rows,
                column_name=self._columns[column_index],
            )
        self._rows += 1
        self._check_time()

    def incomplete(self, reason: str) -> None:
        raise ArtifactError(
            "artifact_incomplete_export",
            "The exact XLSX export could not be completed within its fixed bounds.",
            {
                "reason": reason,
                "completed_rows": self._rows,
                "completed_columns": len(self._columns),
                "completed_bytes": len(self._content),
            },
        )

    def finish(self) -> bytes:
        self._check_open()
        self._check_time()
        columns_json = canonical_json(self._columns)
        provenance_rows: tuple[tuple[str, object], ...] = (
            ("Key", "Value"),
            ("Authorship", ArtifactAuthorship.EXACT_SOURCE_DATA.value),
            ("Source ID", self._provenance.source_id),
            ("Source Revision", self._provenance.source_revision),
            (
                "Resource Revisions",
                canonical_json(self._provenance.resource_revisions),
            ),
            ("SQL Fingerprint", self._provenance.sql_fingerprint),
            ("Parameters SHA-256", self._provenance.parameters_sha256),
            (
                "Columns SHA-256",
                "sha256:" + sha256(columns_json.encode("utf-8")).hexdigest(),
            ),
            ("Column Count", len(self._columns)),
            ("Row Count", self._rows),
            ("Sensitivity", self._provenance.sensitivity.value),
            ("Created At", _utc_z(self._provenance.created_at)),
        )
        for row_index, (key, value) in enumerate(provenance_rows):
            _write_xlsx_text(
                self._provenance_sheet,
                row_index,
                0,
                key,
                row_index=row_index,
                column_name="Key",
            )
            if type(value) is int:
                self._provenance_sheet.write_number(row_index, 1, value)
            else:
                assert isinstance(value, str)
                _write_xlsx_text(
                    self._provenance_sheet,
                    row_index,
                    1,
                    value,
                    row_index=row_index,
                    column_name="Value",
                )
        self._workbook.close()
        self._closed = True
        self._content = self._buffer.getvalue()
        if len(self._content) > self._max_bytes:
            self.incomplete("byte_limit")
        self._check_time()
        try:
            verify_exact_xlsx(
                self._content,
                max_bytes=self._max_bytes,
                max_uncompressed_bytes=self._max_uncompressed_bytes,
                max_members=self._max_members,
            )
        except ArtifactError as error:
            reason = error.details.get("reason")
            raise ArtifactError(
                "artifact_incomplete_export",
                "The exact XLSX package failed bounded verification.",
                {
                    "reason": reason if isinstance(reason, str) else "invalid_package",
                    "completed_rows": self._rows,
                    "completed_columns": len(self._columns),
                    "completed_bytes": len(self._content),
                },
            ) from error
        self._check_time()
        return self._content

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("exact XLSX renderer is already closed")

    def _check_time(self) -> None:
        if self._clock() >= self._deadline:
            self.incomplete("time_limit")


def render_exact_xlsx(
    columns: Sequence[str],
    rows: Iterable[Sequence[object]],
    *,
    provenance: ExactXlsxProvenance,
    max_rows: int = MAX_XLSX_ROWS,
    max_columns: int = MAX_XLSX_COLUMNS,
    max_bytes: int = MAX_XLSX_BYTES,
    max_uncompressed_bytes: int = MAX_XLSX_UNCOMPRESSED_BYTES,
    max_members: int = MAX_XLSX_MEMBERS,
    max_seconds: float = MAX_XLSX_SECONDS,
    clock: Callable[[], float] = time_module.monotonic,
) -> bytes:
    """Render an exhausted typed iterable as one verified deterministic workbook."""

    renderer = ExactXlsxRenderer(
        columns,
        provenance=provenance,
        max_rows=max_rows,
        max_columns=max_columns,
        max_bytes=max_bytes,
        max_uncompressed_bytes=max_uncompressed_bytes,
        max_members=max_members,
        max_seconds=max_seconds,
        clock=clock,
    )
    for row in rows:
        renderer.append(row)
    return renderer.finish()


def verify_exact_xlsx(
    content: bytes,
    *,
    max_bytes: int = MAX_XLSX_BYTES,
    max_uncompressed_bytes: int = MAX_XLSX_UNCOMPRESSED_BYTES,
    max_members: int = MAX_XLSX_MEMBERS,
) -> None:
    """Fail closed unless bytes match Daita's fixed safe XLSX package."""

    if not isinstance(content, bytes) or not content:
        _invalid_xlsx("invalid_package", completed_bytes=0)
    if len(content) > max_bytes:
        _invalid_xlsx("byte_limit", completed_bytes=len(content))
    try:
        with ZipFile(BytesIO(content), "r") as archive:
            infos = archive.infolist()
            names = tuple(item.filename for item in infos)
            if len(infos) > max_members:
                _invalid_xlsx("member_limit", completed_bytes=len(content))
            if len(names) != len(set(names)):
                _invalid_xlsx("duplicate_member", completed_bytes=len(content))
            if names != _XLSX_MEMBER_ORDER:
                _invalid_xlsx("unexpected_members", completed_bytes=len(content))
            if archive.comment:
                _invalid_xlsx("archive_comment", completed_bytes=len(content))
            uncompressed = 0
            parsed: dict[str, ElementTree.Element] = {}
            for info in infos:
                if _unsafe_zip_name(info.filename):
                    _invalid_xlsx("unsafe_member_name", completed_bytes=len(content))
                if info.is_dir() or info.compress_type != ZIP_DEFLATED:
                    _invalid_xlsx("invalid_member_type", completed_bytes=len(content))
                if info.flag_bits & 0x1:
                    _invalid_xlsx("encrypted_member", completed_bytes=len(content))
                if info.date_time != _XLSX_TIMESTAMP or info.extra or info.comment:
                    _invalid_xlsx(
                        "nondeterministic_member", completed_bytes=len(content)
                    )
                mode = info.external_attr >> 16
                if stat.S_IFMT(mode) == stat.S_IFLNK:
                    _invalid_xlsx("symlink_member", completed_bytes=len(content))
                uncompressed += info.file_size
                if uncompressed > max_uncompressed_bytes:
                    _invalid_xlsx(
                        "uncompressed_byte_limit", completed_bytes=len(content)
                    )
                raw = archive.read(info)
                if info.filename.endswith((".xml", ".rels")):
                    upper = raw.upper()
                    if b"<!DOCTYPE" in upper or b"<!ENTITY" in upper:
                        _invalid_xlsx("unsafe_xml", completed_bytes=len(content))
                    try:
                        parsed[info.filename] = ElementTree.fromstring(raw)
                    except ElementTree.ParseError:
                        _invalid_xlsx("invalid_xml", completed_bytes=len(content))
    except (BadZipFile, OSError, RuntimeError, ValueError) as error:
        if isinstance(error, ArtifactError):
            raise
        raise ArtifactError(
            "artifact_incomplete_export",
            "The exact XLSX package failed bounded verification.",
            {
                "reason": "invalid_package",
                "completed_rows": 0,
                "completed_columns": 0,
                "completed_bytes": len(content) if isinstance(content, bytes) else 0,
            },
        ) from error

    workbook = parsed["xl/workbook.xml"]
    sheets = workbook.find(f"{{{_SPREADSHEET_NAMESPACE}}}sheets")
    sheet_nodes = () if sheets is None else tuple(sheets)
    if tuple(node.get("name") for node in sheet_nodes) != ("Data", "Provenance"):
        _invalid_xlsx("invalid_sheets", completed_bytes=len(content))
    if any(node.get("state", "visible") != "visible" for node in sheet_nodes):
        _invalid_xlsx("hidden_sheet", completed_bytes=len(content))
    if workbook.find(f"{{{_SPREADSHEET_NAMESPACE}}}definedNames") is not None:
        _invalid_xlsx("defined_names", completed_bytes=len(content))

    for name in ("xl/worksheets/sheet1.xml", "xl/worksheets/sheet2.xml"):
        for node in parsed[name].iter():
            local_name = node.tag.rsplit("}", 1)[-1]
            if local_name == "f" or local_name in _PROHIBITED_WORKSHEET_ELEMENTS:
                _invalid_xlsx("unsafe_worksheet", completed_bytes=len(content))

    for name, expected in _XLSX_RELATIONSHIPS.items():
        relationships = parsed[name].findall(
            f"{{{_RELATIONSHIP_NAMESPACE}}}Relationship"
        )
        actual = tuple(
            (relationship.get("Type"), relationship.get("Target"))
            for relationship in relationships
        )
        for relationship in relationships:
            relation_type = relationship.get("Type", "").casefold()
            if relationship.get("TargetMode", "").casefold() == "external" or any(
                prohibited in relation_type
                for prohibited in ("hyperlink", "externallink", "oleobject")
            ):
                _invalid_xlsx("external_relationship", completed_bytes=len(content))
        if actual != expected:
            _invalid_xlsx("unexpected_relationship", completed_bytes=len(content))

    content_types = parsed["[Content_Types].xml"]
    for node in content_types:
        content_type = node.get("ContentType", "").casefold()
        if any(
            prohibited in content_type
            for prohibited in ("macro", "vba", "activex", "oleobject")
        ):
            _invalid_xlsx("unsafe_content_type", completed_bytes=len(content))

    core = parsed["docProps/core.xml"]
    if (
        core.findtext(f"{{{_DC_NAMESPACE}}}title") != "Daita exact tabular export"
        or core.findtext(f"{{{_DC_NAMESPACE}}}subject") != "Exact source data"
        or core.findtext(f"{{{_DC_NAMESPACE}}}creator") != "Daita"
        or core.findtext(f"{{{_DC_NAMESPACE}}}description")
        != "Deterministic literal-only workbook"
        or core.findtext(f"{{{_CORE_NAMESPACE}}}lastModifiedBy") != "Daita"
    ):
        _invalid_xlsx("workbook_properties", completed_bytes=len(content))
    created = core.findtext(f"{{{_DCTERMS_NAMESPACE}}}created")
    modified = core.findtext(f"{{{_DCTERMS_NAMESPACE}}}modified")
    if (
        created != modified
        or not isinstance(created, str)
        or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", created)
    ):
        _invalid_xlsx("workbook_time", completed_bytes=len(content))

    app = parsed["docProps/app.xml"]
    if (
        app.findtext(f"{{{_EXTENDED_PROPERTIES_NAMESPACE}}}Application")
        != "Microsoft Excel"
        or app.findtext(f"{{{_EXTENDED_PROPERTIES_NAMESPACE}}}Company") != "Daita"
        or app.findtext(f"{{{_EXTENDED_PROPERTIES_NAMESPACE}}}DocSecurity") != "0"
    ):
        _invalid_xlsx("extended_properties", completed_bytes=len(content))

    styles = parsed["xl/styles.xml"]
    number_formats = styles.find(f"{{{_SPREADSHEET_NAMESPACE}}}numFmts")
    format_codes = (
        ()
        if number_formats is None
        else tuple(item.get("formatCode") for item in number_formats)
    )
    if format_codes not in {(), ("yyyy-mm-dd",)}:
        _invalid_xlsx("locale_sensitive_format", completed_bytes=len(content))


def _load_xlsxwriter() -> Any:
    try:
        module = import_module("xlsxwriter")
    except (ImportError, AttributeError) as error:
        raise ImportError(
            "Daita's XLSX runtime dependency is unavailable. " f"{PIPX_REPAIR_GUIDANCE}"
        ) from error
    if not callable(getattr(module, "Workbook", None)) or not str(
        getattr(module, "__version__", "")
    ).startswith("3."):
        raise ImportError(
            "Daita's XLSX runtime dependency is unavailable. " f"{PIPX_REPAIR_GUIDANCE}"
        )
    return module


def _write_xlsx_scalar(
    worksheet: Any,
    row: int,
    column: int,
    value: object,
    *,
    date_format: Any,
    row_index: int,
    column_name: str,
) -> None:
    if value is None:
        return
    if type(value) is str:
        _write_xlsx_text(
            worksheet,
            row,
            column,
            value,
            row_index=row_index,
            column_name=column_name,
        )
        return
    if type(value) is bool:
        worksheet.write_boolean(row, column, value)
        return
    if type(value) is int:
        digits = len(str(abs(value)))
        if digits <= 15 and abs(value) <= 9_007_199_254_740_991:
            worksheet.write_number(row, column, value)
        else:
            _write_xlsx_text(
                worksheet,
                row,
                column,
                str(value),
                row_index=row_index,
                column_name=column_name,
            )
        return
    if type(value) is float:
        if math.isfinite(value):
            worksheet.write_number(row, column, value)
            return
        raise _unsupported_xlsx_value(value, row_index, column, column_name)
    if type(value) is Decimal:
        if value.is_finite():
            _write_xlsx_text(
                worksheet,
                row,
                column,
                format(value, "f"),
                row_index=row_index,
                column_name=column_name,
            )
            return
        raise _unsupported_xlsx_value(value, row_index, column, column_name)
    if type(value) is datetime:
        _write_xlsx_text(
            worksheet,
            row,
            column,
            value.isoformat(timespec="microseconds"),
            row_index=row_index,
            column_name=column_name,
        )
        return
    if type(value) is date:
        worksheet.write_datetime(row, column, value, date_format)
        return
    if type(value) is time:
        _write_xlsx_text(
            worksheet,
            row,
            column,
            value.isoformat(timespec="microseconds"),
            row_index=row_index,
            column_name=column_name,
        )
        return
    if type(value) in {bytes, bytearray, memoryview}:
        binary = bytes(cast(bytes | bytearray | memoryview, value))
        _write_xlsx_text(
            worksheet,
            row,
            column,
            r"\B" + base64.b64encode(binary).decode("ascii"),
            row_index=row_index,
            column_name=column_name,
        )
        return
    if type(value) is UUID:
        _write_xlsx_text(
            worksheet,
            row,
            column,
            str(value).lower(),
            row_index=row_index,
            column_name=column_name,
        )
        return
    raise _unsupported_xlsx_value(value, row_index, column, column_name)


def _write_xlsx_text(
    worksheet: Any,
    row: int,
    column: int,
    value: str,
    *,
    row_index: int,
    column_name: str,
) -> None:
    try:
        units = len(value.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        raise _unsupported_xlsx_value(value, row_index, column, column_name) from None
    if units > MAX_XLSX_TEXT_UTF16_UNITS:
        raise _unsupported_xlsx_value(value, row_index, column, column_name)
    worksheet.write_string(row, column, value)


def _unsupported_xlsx_value(
    value: object,
    row_index: int,
    column_index: int,
    column_name: str,
    *,
    message: str = (
        "The source returned a value without a lossless exact XLSX representation."
    ),
) -> ArtifactError:
    return ArtifactError(
        "artifact_unsupported_value",
        message,
        {
            "row_index": row_index,
            "column_index": column_index,
            "column_name": column_name,
            "runtime_type": _safe_runtime_type(value),
        },
    )


def _invalid_xlsx(reason: str, *, completed_bytes: int) -> None:
    raise ArtifactError(
        "artifact_incomplete_export",
        "The exact XLSX package failed bounded verification.",
        {
            "reason": reason,
            "completed_rows": 0,
            "completed_columns": 0,
            "completed_bytes": completed_bytes,
        },
    )


def _unsafe_zip_name(value: str) -> bool:
    if not value or value.startswith(("/", "\\")) or "\\" in value:
        return True
    parts = value.split("/")
    return any(part in {"", ".", ".."} for part in parts)


def _utc_z(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


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
    return _validated_tabular_columns(columns, max_columns, format_name="CSV")


def _validated_tabular_columns(
    columns: tuple[str, ...],
    max_columns: int,
    *,
    format_name: str,
) -> tuple[str, ...]:
    if not columns:
        raise ArtifactError(
            "artifact_unsupported_value",
            f"The exact {format_name} result must contain at least one named column.",
            {"column_index": 0, "reason": "missing_columns", "runtime_type": "str"},
        )
    if len(columns) > max_columns:
        raise ArtifactError(
            "artifact_quota_exceeded",
            f"The exact {format_name} result exceeds its column limit.",
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
                f"The exact {format_name} result contains an unsupported column name.",
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
    "ExactXlsxProvenance",
    "ExactXlsxRenderer",
    "MAX_CSV_BYTES",
    "MAX_CSV_COLUMNS",
    "MAX_CSV_ROWS",
    "MAX_CSV_SECONDS",
    "MAX_XLSX_BYTES",
    "MAX_XLSX_COLUMNS",
    "MAX_XLSX_MEMBERS",
    "MAX_XLSX_ROWS",
    "MAX_XLSX_SECONDS",
    "MAX_XLSX_TEXT_UTF16_UNITS",
    "MAX_XLSX_UNCOMPRESSED_BYTES",
    "XLSX_ALLOWED_EXTENSIONS",
    "XLSX_MEDIA_TYPE",
    "render_exact_csv",
    "render_exact_xlsx",
    "render_model_document",
    "verify_exact_xlsx",
]
