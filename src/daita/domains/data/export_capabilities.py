"""Fixed document, exact-CSV, and local-delivery capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from hashlib import sha256
from typing import Protocol

from ..._json import FrozenJsonObject, canonical_json
from ...capabilities import (
    AccessMode,
    ArtifactPolicy,
    Capability,
    Executor,
    ExtensionDeclarations,
    ToolApplicability,
    ToolExecution,
    ToolOutput,
    ToolView,
)
from ...artifacts.delivery import LocalArtifactDelivery
from ...artifacts.models import (
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
    ArtifactProvenance,
    ArtifactResourceBinding,
    MAX_ARTIFACT_BYTES,
    MAX_DOCUMENT_BYTES,
    artifact_delivery_receipt_to_mapping,
    artifact_destination_to_mapping,
    canonical_artifact_filename,
)
from ...artifacts.renderers import (
    CSV_ALLOWED_EXTENSIONS,
    DOCUMENT_ALLOWED_EXTENSIONS,
    MAX_CSV_BYTES,
    MAX_CSV_COLUMNS,
    MAX_CSV_ROWS,
    MAX_CSV_SECONDS,
    render_model_document,
)
from ...catalog.models import Sensitivity

DOCUMENT_CREATE_CAPABILITY_ID = "artifact.create_document"
DOCUMENT_CREATE_EXECUTOR_ID = "artifact.create_document.executor"
DOCUMENT_CREATE_TOOL_NAME = "artifact_create_document"
DOCUMENT_CREATE_OUTPUT_KIND = "artifact.document"

SQLITE_CSV_EXPORT_CAPABILITY_ID = "data.sqlite.export_csv"
SQLITE_CSV_EXPORT_EXECUTOR_ID = "data.sqlite.export_csv.executor"
SQLITE_CSV_EXPORT_TOOL_NAME = "data_export_sqlite"
POSTGRESQL_CSV_EXPORT_CAPABILITY_ID = "data.postgresql.export_csv"
POSTGRESQL_CSV_EXPORT_EXECUTOR_ID = "data.postgresql.export_csv.executor"
POSTGRESQL_CSV_EXPORT_TOOL_NAME = "data_export_postgresql"
CSV_EXPORT_OUTPUT_KIND = "artifact.csv_export"

ARTIFACT_SAVE_LOCAL_CAPABILITY_ID = "artifact.save_local"
ARTIFACT_SAVE_LOCAL_EXECUTOR_ID = "artifact.save_local.executor"
ARTIFACT_SAVE_LOCAL_TOOL_NAME = "artifact_save_local"
ARTIFACT_DELIVERY_RECEIPT_OUTPUT_KIND = "artifact.delivery_receipt"

ARTIFACT_SET_EXPORT_LOCATION_CAPABILITY_ID = "artifact.set_export_location"
ARTIFACT_SET_EXPORT_LOCATION_EXECUTOR_ID = "artifact.set_export_location.executor"
ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME = "artifact_set_export_location"
ARTIFACT_EXPORT_LOCATION_OUTPUT_KIND = "artifact.export_location"


@dataclass(frozen=True, slots=True)
class ArtifactCapabilityDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


@dataclass(frozen=True, slots=True)
class ExactCsvExportResult:
    """One complete typed-source export after deterministic CSV rendering."""

    source_id: str
    source_revision: str
    sql_fingerprint: str
    resource_revisions: tuple[tuple[str, str], ...]
    columns: tuple[str, ...]
    row_count: int
    content: bytes

    def __post_init__(self) -> None:
        for value, name in (
            (self.source_id, "source_id"),
            (self.source_revision, "source_revision"),
            (self.sql_fingerprint, "sql_fingerprint"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"exact CSV {name} must be non-empty text")
        if not self.sql_fingerprint.startswith("sha256:"):
            raise ValueError("exact CSV sql_fingerprint must use sha256")
        revisions = tuple(sorted(tuple(item) for item in self.resource_revisions))
        if not revisions or len(revisions) > 64:
            raise ValueError("exact CSV resource revisions exceed their bound")
        if len({item[0] for item in revisions}) != len(revisions) or any(
            not resource_id or not revision.startswith("sha256:")
            for resource_id, revision in revisions
        ):
            raise ValueError("exact CSV resource revisions are invalid")
        columns = tuple(self.columns)
        if (
            not 1 <= len(columns) <= MAX_CSV_COLUMNS
            or len(columns) != len(set(columns))
            or any(
                not isinstance(column, str) or not column.strip() or len(column) > 256
                for column in columns
            )
        ):
            raise ValueError("exact CSV columns are invalid")
        if (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or not 0 <= self.row_count <= MAX_CSV_ROWS
        ):
            raise ValueError("exact CSV row_count is outside its bound")
        if (
            not isinstance(self.content, bytes)
            or not self.content
            or len(self.content) > MAX_CSV_BYTES
        ):
            raise ValueError("exact CSV content is outside its byte bound")
        object.__setattr__(self, "resource_revisions", revisions)
        object.__setattr__(self, "columns", columns)


ExactCsvProgress = Callable[[int, int, int], None]


class ExactCsvExportBackend(Protocol):
    async def execute_exact_csv(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        max_rows: int,
        max_columns: int,
        max_bytes: int,
        timeout_seconds: float,
        progress: ExactCsvProgress | None = None,
    ) -> ExactCsvExportResult: ...


class DocumentArtifactExecutor:
    executor_id = DOCUMENT_CREATE_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        content = request.arguments["content"]
        format_name = request.arguments["format"]
        filename = request.arguments.get("filename")
        evidence = request.arguments.get("evidence_call_ids", ())
        assert isinstance(content, str)
        assert isinstance(format_name, str)
        assert filename is None or isinstance(filename, str)
        assert isinstance(evidence, tuple)
        draft = render_model_document(
            content=content,
            format=format_name,
            filename=filename,
            evidence_call_ids=tuple(str(item) for item in evidence),
        )
        return ToolOutput(
            kind=DOCUMENT_CREATE_OUTPUT_KIND,
            data={"format": format_name, "character_count": len(content)},
            artifact=draft,
        )


class _CsvExportExecutor:
    executor_id: str

    def __init__(
        self,
        agent_id: str,
        backend: ExactCsvExportBackend,
        *,
        max_seconds: float = MAX_CSV_SECONDS,
    ) -> None:
        self._agent_id = agent_id
        self._backend = backend
        self._max_seconds = max_seconds

    async def execute(self, request: ToolExecution) -> ToolOutput:
        source_id = request.arguments["source_id"]
        sql = request.arguments["sql"]
        parameters = request.arguments.get("parameters", ())
        format_name = request.arguments["format"]
        filename = request.arguments.get("filename")
        assert isinstance(source_id, str)
        assert isinstance(sql, str)
        assert isinstance(parameters, tuple)
        assert format_name == "csv"
        assert filename is None or isinstance(filename, str)
        completed = {"rows": 0, "columns": 0, "bytes": 0}

        def progress(rows: int, columns: int, byte_count: int) -> None:
            completed.update(rows=rows, columns=columns, bytes=byte_count)

        try:
            async with asyncio.timeout(self._max_seconds):
                result = await self._backend.execute_exact_csv(
                    agent_id=self._agent_id,
                    source_id=source_id,
                    sql=sql,
                    parameters=parameters,
                    max_rows=MAX_CSV_ROWS,
                    max_columns=MAX_CSV_COLUMNS,
                    max_bytes=MAX_CSV_BYTES,
                    timeout_seconds=self._max_seconds,
                    progress=progress,
                )
        except TimeoutError as error:
            raise ArtifactError(
                "artifact_incomplete_export",
                "The exact CSV export exceeded its execution-time limit.",
                {
                    "reason": "time_limit",
                    "completed_rows": completed["rows"],
                    "completed_columns": completed["columns"],
                    "completed_bytes": completed["bytes"],
                },
            ) from error
        if result.source_id != source_id:
            raise ValueError("exact CSV backend returned a different source")
        safe_filename = canonical_artifact_filename(
            filename or "export.csv",
            "text/csv",
            CSV_ALLOWED_EXTENSIONS,
        )
        bindings = tuple(
            ArtifactResourceBinding(
                source_id=result.source_id,
                source_revision=result.source_revision,
                resource_id=resource_id,
                resource_revision=revision,
            )
            for resource_id, revision in result.resource_revisions
        )
        parameters_sha256 = (
            "sha256:" + sha256(canonical_json(parameters).encode("utf-8")).hexdigest()
        )
        draft = ArtifactDraft(
            content=result.content,
            suggested_filename=safe_filename,
            media_type="text/csv",
            sensitivity=Sensitivity.PUBLIC,
            provenance=ArtifactProvenance(
                authorship=ArtifactAuthorship.EXACT_SOURCE_DATA,
                resource_bindings=bindings,
                sql_fingerprint=result.sql_fingerprint,
                parameters_sha256=parameters_sha256,
                columns=result.columns,
                row_count=result.row_count,
            ),
        )
        return ToolOutput(
            kind=CSV_EXPORT_OUTPUT_KIND,
            data={
                "format": "csv",
                "filename": safe_filename,
                "row_count": result.row_count,
                "column_count": len(result.columns),
            },
            artifact=draft,
        )


class SQLiteCsvExportExecutor(_CsvExportExecutor):
    executor_id = SQLITE_CSV_EXPORT_EXECUTOR_ID


class PostgreSQLCsvExportExecutor(_CsvExportExecutor):
    executor_id = POSTGRESQL_CSV_EXPORT_EXECUTOR_ID


class ArtifactSaveLocalExecutor:
    executor_id = ARTIFACT_SAVE_LOCAL_EXECUTOR_ID

    def __init__(self, delivery: LocalArtifactDelivery) -> None:
        self._delivery = delivery

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        artifact_id = request.arguments["artifact_id"]
        destination_id = request.arguments["destination_id"]
        filename = request.arguments.get("filename")
        assert isinstance(artifact_id, str)
        assert isinstance(destination_id, str)
        assert filename is None or isinstance(filename, str)
        return await self._delivery.preflight_save(
            run_id=request.run_id,
            artifact_id=artifact_id,
            destination_id=destination_id,
            filename=filename,
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        artifact_id = request.arguments["artifact_id"]
        destination_id = request.arguments["destination_id"]
        filename = request.arguments.get("filename")
        assert isinstance(artifact_id, str)
        assert isinstance(destination_id, str)
        assert filename is None or isinstance(filename, str)
        receipt = await self._delivery.save_committed(
            run_id=request.run_id,
            artifact_id=artifact_id,
            destination_id=destination_id,
            filename=filename,
        )
        return ToolOutput(
            kind=ARTIFACT_DELIVERY_RECEIPT_OUTPUT_KIND,
            data=artifact_delivery_receipt_to_mapping(receipt),
        )


class ArtifactSetExportLocationExecutor:
    executor_id = ARTIFACT_SET_EXPORT_LOCATION_EXECUTOR_ID

    def __init__(self, delivery: LocalArtifactDelivery) -> None:
        self._delivery = delivery

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        destination_id = request.arguments["destination_id"]
        assert isinstance(destination_id, str)
        return await self._delivery.preflight_set_default(destination_id)

    async def execute(self, request: ToolExecution) -> ToolOutput:
        destination_id = request.arguments["destination_id"]
        assert isinstance(destination_id, str)
        destination = await self._delivery.set_default_by_id(destination_id)
        return ToolOutput(
            kind=ARTIFACT_EXPORT_LOCATION_OUTPUT_KIND,
            data=artifact_destination_to_mapping(destination),
        )


def artifact_capability_declarations(
    delivery: LocalArtifactDelivery,
    *,
    agent_id: str,
    sqlite_backend: ExactCsvExportBackend,
    postgresql_backend: ExactCsvExportBackend,
) -> ArtifactCapabilityDeclarations:
    extension = artifact_extension_declarations()
    return ArtifactCapabilityDeclarations(
        capabilities=extension.capabilities,
        executors=(
            DocumentArtifactExecutor(),
            SQLiteCsvExportExecutor(agent_id, sqlite_backend),
            PostgreSQLCsvExportExecutor(agent_id, postgresql_backend),
            ArtifactSaveLocalExecutor(delivery),
            ArtifactSetExportLocationExecutor(delivery),
        ),
        tool_views=extension.tool_views,
    )


def artifact_extension_declarations() -> ExtensionDeclarations:
    document = Capability(
        id=DOCUMENT_CREATE_CAPABILITY_ID,
        description=(
            "Package one bounded model-authored Markdown or TXT document as a "
            "durable internal artifact."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["markdown", "txt"]},
                "filename": {"type": "string", "minLength": 1, "maxLength": 120},
                "content": {"type": "string", "minLength": 1, "maxLength": 200_000},
                "evidence_call_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1, "maxLength": 256},
                    "maxItems": 16,
                    "uniqueItems": True,
                },
            },
            "required": ["format", "content"],
            "additionalProperties": False,
        },
        output_kind=DOCUMENT_CREATE_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "format": {"type": "string"},
                "character_count": {"type": "integer"},
            },
            "required": ["format", "character_count"],
            "additionalProperties": False,
        },
        executor_id=DOCUMENT_CREATE_EXECUTOR_ID,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({"text/markdown", "text/plain"}),
            allowed_extensions=DOCUMENT_ALLOWED_EXTENSIONS,
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=MAX_DOCUMENT_BYTES,
            max_total_bytes_per_call=MAX_DOCUMENT_BYTES,
        ),
    )
    sqlite_csv = _csv_export_capability(
        capability_id=SQLITE_CSV_EXPORT_CAPABILITY_ID,
        executor_id=SQLITE_CSV_EXPORT_EXECUTOR_ID,
        adapter_name="SQLite",
    )
    postgresql_csv = _csv_export_capability(
        capability_id=POSTGRESQL_CSV_EXPORT_CAPABILITY_ID,
        executor_id=POSTGRESQL_CSV_EXPORT_EXECUTOR_ID,
        adapter_name="PostgreSQL",
    )
    save = Capability(
        id=ARTIFACT_SAVE_LOCAL_CAPABILITY_ID,
        description=(
            "Copy one committed artifact to the effective default or an exact "
            "authorized local destination without overwrite."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "pattern": "^artifact-[0-9a-f]{32}$",
                },
                "destination_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                },
                "filename": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 120,
                },
            },
            "required": ["artifact_id", "destination_id"],
            "additionalProperties": False,
        },
        output_kind=ARTIFACT_DELIVERY_RECEIPT_OUTPUT_KIND,
        output_schema=_receipt_schema(),
        executor_id=ARTIFACT_SAVE_LOCAL_EXECUTOR_ID,
        access_mode=AccessMode.WRITE,
        side_effecting=True,
    )
    set_location = Capability(
        id=ARTIFACT_SET_EXPORT_LOCATION_CAPABILITY_ID,
        description=(
            "Set one already authorized persistent destination as the default for "
            "future artifact exports."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "destination_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                }
            },
            "required": ["destination_id"],
            "additionalProperties": False,
        },
        output_kind=ARTIFACT_EXPORT_LOCATION_OUTPUT_KIND,
        output_schema=_destination_schema(),
        executor_id=ARTIFACT_SET_EXPORT_LOCATION_EXECUTOR_ID,
        access_mode=AccessMode.WRITE,
        side_effecting=True,
    )
    capabilities = (document, sqlite_csv, postgresql_csv, save, set_location)
    views = (
        ToolView(
            name=DOCUMENT_CREATE_TOOL_NAME,
            capability_id=document.id,
            description=document.description,
        ),
        ToolView(
            name=SQLITE_CSV_EXPORT_TOOL_NAME,
            capability_id=sqlite_csv.id,
            description=sqlite_csv.description,
            applicability=ToolApplicability(
                source_adapter_ids=("sqlite",), minimum_active_sources=1
            ),
        ),
        ToolView(
            name=POSTGRESQL_CSV_EXPORT_TOOL_NAME,
            capability_id=postgresql_csv.id,
            description=postgresql_csv.description,
            applicability=ToolApplicability(
                source_adapter_ids=("postgresql",), minimum_active_sources=1
            ),
        ),
        ToolView(
            name=ARTIFACT_SAVE_LOCAL_TOOL_NAME,
            capability_id=save.id,
            description=save.description,
        ),
        ToolView(
            name=ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
            capability_id=set_location.id,
            description=set_location.description,
        ),
    )
    return ExtensionDeclarations(
        capabilities=capabilities,
        executor_ids=tuple(item.executor_id for item in capabilities),
        tool_views=views,
    )


def _csv_export_capability(
    *,
    capability_id: str,
    executor_id: str,
    adapter_name: str,
) -> Capability:
    return Capability(
        id=capability_id,
        description=(
            f"Run one fresh validated read-only {adapter_name} query and create one "
            "complete exact CSV artifact without routing rows through the model."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "minLength": 1, "maxLength": 256},
                "sql": {"type": "string", "minLength": 1},
                "parameters": {"type": "array"},
                "format": {"type": "string", "enum": ["csv"]},
                "filename": {"type": "string", "minLength": 1, "maxLength": 120},
            },
            "required": ["source_id", "sql", "format"],
            "additionalProperties": False,
        },
        output_kind=CSV_EXPORT_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["csv"]},
                "filename": {"type": "string"},
                "row_count": {"type": "integer", "minimum": 0},
                "column_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_CSV_COLUMNS,
                },
            },
            "required": ["format", "filename", "row_count", "column_count"],
            "additionalProperties": False,
        },
        executor_id=executor_id,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({"text/csv"}),
            allowed_extensions=CSV_ALLOWED_EXTENSIONS,
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=MAX_ARTIFACT_BYTES,
            max_total_bytes_per_call=MAX_ARTIFACT_BYTES,
        ),
    )


def _receipt_schema() -> dict[str, object]:
    types = {
        "artifact_id": "string",
        "destination_id": "string",
        "filename": "string",
        "saved_path": "string",
        "byte_size": "integer",
        "sha256": "string",
        "renamed_for_collision": "boolean",
        "delivered_at": "string",
    }
    return {
        "type": "object",
        "properties": {name: {"type": kind} for name, kind in types.items()},
        "required": list(types),
        "additionalProperties": False,
    }


def _destination_schema() -> dict[str, object]:
    types = {
        "destination_id": "string",
        "display_name": "string",
        "kind": "string",
        "authorization": "string",
        "availability": "string",
        "is_default": "boolean",
    }
    return {
        "type": "object",
        "properties": {name: {"type": kind} for name, kind in types.items()},
        "required": list(types),
        "additionalProperties": False,
    }


__all__ = [
    "ARTIFACT_DELIVERY_RECEIPT_OUTPUT_KIND",
    "ARTIFACT_EXPORT_LOCATION_OUTPUT_KIND",
    "ARTIFACT_SAVE_LOCAL_CAPABILITY_ID",
    "ARTIFACT_SAVE_LOCAL_EXECUTOR_ID",
    "ARTIFACT_SAVE_LOCAL_TOOL_NAME",
    "ARTIFACT_SET_EXPORT_LOCATION_CAPABILITY_ID",
    "ARTIFACT_SET_EXPORT_LOCATION_EXECUTOR_ID",
    "ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME",
    "CSV_EXPORT_OUTPUT_KIND",
    "DOCUMENT_CREATE_CAPABILITY_ID",
    "DOCUMENT_CREATE_EXECUTOR_ID",
    "DOCUMENT_CREATE_OUTPUT_KIND",
    "DOCUMENT_CREATE_TOOL_NAME",
    "ExactCsvExportBackend",
    "ExactCsvProgress",
    "ExactCsvExportResult",
    "POSTGRESQL_CSV_EXPORT_CAPABILITY_ID",
    "POSTGRESQL_CSV_EXPORT_EXECUTOR_ID",
    "POSTGRESQL_CSV_EXPORT_TOOL_NAME",
    "PostgreSQLCsvExportExecutor",
    "SQLITE_CSV_EXPORT_CAPABILITY_ID",
    "SQLITE_CSV_EXPORT_EXECUTOR_ID",
    "SQLITE_CSV_EXPORT_TOOL_NAME",
    "SQLiteCsvExportExecutor",
    "artifact_capability_declarations",
    "artifact_extension_declarations",
]
