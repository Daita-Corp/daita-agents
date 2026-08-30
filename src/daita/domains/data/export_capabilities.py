"""Declare and execute artifact creation, conversion, continuity, and delivery tools."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from hashlib import sha256
from typing import Protocol

from ..._json import FrozenJsonObject, canonical_json
from ...adapters.local_workspace import LocalWorkspaceBackend, LocalWorkspaceError
from ...artifacts.delivery import LocalArtifactDelivery
from ...artifacts.models import (
    MAX_DOCUMENT_BYTES,
    MAX_TEXT_EDIT_OPERATIONS,
    ArtifactAuthorship,
    ArtifactDraft,
    ArtifactError,
    ArtifactLocalFileBinding,
    ArtifactPayload,
    ArtifactProvenance,
    ArtifactRef,
    ArtifactResourceBinding,
    ArtifactResultBinding,
    artifact_delivery_receipt_to_mapping,
    artifact_destination_to_mapping,
    artifact_text_change_summary_to_mapping,
    canonical_artifact_filename,
)
from ...artifacts.renderers import (
    CSV_ALLOWED_EXTENSIONS,
    DOCUMENT_ALLOWED_EXTENSIONS,
    MAX_CSV_BYTES,
    MAX_CSV_COLUMNS,
    MAX_CSV_ROWS,
    MAX_CSV_SECONDS,
    MAX_TEXT_EDIT_ANCHOR_BYTES,
    MAX_TEXT_EDIT_BYTES,
    MAX_TEXT_EDIT_OCCURRENCES,
    MAX_TEXT_EDIT_REPLACEMENT_BYTES,
    MAX_XLSX_BYTES,
    MAX_XLSX_SECONDS,
    TEXT_EDIT_ALLOWED_EXTENSIONS,
    TEXT_EDIT_MEDIA_TYPES,
    XLSX_ALLOWED_EXTENSIONS,
    XLSX_MEDIA_TYPE,
    apply_bounded_text_edits,
    read_exact_xlsx_data,
    render_exact_csv,
    render_model_document,
    text_edit_media_type,
)
from ...artifacts.result_snapshot import (
    MAX_RESULT_SNAPSHOT_BYTES,
    RESULT_SNAPSHOT_ALLOWED_EXTENSIONS,
    RESULT_SNAPSHOT_MEDIA_TYPE,
    serialize_result_snapshot,
)
from ...artifacts.store import AgentHomeArtifactStore
from ...capabilities import (
    AccessMode,
    ArtifactPolicy,
    AutomationEligibility,
    Capability,
    CapabilityDeclarations,
    CapabilityInputError,
    CapabilityRegistry,
    Executor,
    OperationalEffect,
    ToolboxId,
    ToolExecution,
    ToolLoadMode,
    ToolOutput,
    ToolOutputValidationError,
    ToolPresentation,
    ToolTextTrust,
    ToolView,
)
from ...capability_runtime import CapabilityFailure, SideEffectPlan
from ...catalog.models import Sensitivity
from ...llm.models import MessageRole, ModelSensitivity, ToolCall, ToolResultBlock
from ...loop.models import RunInput, RunOrigin, Transcript
from ...storage.sqlite_records import SourcePermissionStateError
from ..learning import LearningCandidateGuard
from .sql import (
    MAX_SQL_CHARACTERS,
    MAX_SQL_PARAMETERS,
    ResourceSchema,
    validate_postgresql_read,
    validate_sqlite_read,
)

DOCUMENT_CREATE_CAPABILITY_ID = "artifact.create_document"
DOCUMENT_CREATE_EXECUTOR_ID = "artifact.create_document.executor"
DOCUMENT_CREATE_TOOL_NAME = "artifact_create_document"
DOCUMENT_CREATE_OUTPUT_KIND = "artifact.document"

RESULT_SNAPSHOT_CAPABILITY_ID = "artifact.snapshot_result"
RESULT_SNAPSHOT_EXECUTOR_ID = "artifact.snapshot_result.executor"
RESULT_SNAPSHOT_TOOL_NAME = "artifact_snapshot_result"
RESULT_SNAPSHOT_OUTPUT_KIND = "artifact.result_snapshot"

SQLITE_TABULAR_EXPORT_CAPABILITY_ID = "data.sqlite.export_tabular"
SQLITE_TABULAR_EXPORT_EXECUTOR_ID = "data.sqlite.export_tabular.executor"
SQLITE_TABULAR_EXPORT_TOOL_NAME = "data_export_sqlite"
POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID = "data.postgresql.export_tabular"
POSTGRESQL_TABULAR_EXPORT_EXECUTOR_ID = "data.postgresql.export_tabular.executor"
POSTGRESQL_TABULAR_EXPORT_TOOL_NAME = "data_export_postgresql"
TABULAR_EXPORT_OUTPUT_KIND = "artifact.tabular_export"

ARTIFACT_LIST_CAPABILITY_ID = "artifact.list"
ARTIFACT_LIST_EXECUTOR_ID = "artifact.list.executor"
ARTIFACT_LIST_TOOL_NAME = "artifact_list"
ARTIFACT_LIST_OUTPUT_KIND = "artifact.list_result"
MAX_MODEL_ARTIFACT_LIST_ITEMS = 50

ARTIFACT_READ_CAPABILITY_ID = "artifact.read"
ARTIFACT_READ_EXECUTOR_ID = "artifact.read.executor"
ARTIFACT_READ_TOOL_NAME = "artifact_read"
ARTIFACT_READ_OUTPUT_KIND = "artifact.read_result"
MAX_MODEL_ARTIFACT_TEXT_BYTES = 16 * 1024
MAX_MODEL_ARTIFACT_XLSX_ROWS = 50

ARTIFACT_CONVERT_CAPABILITY_ID = "artifact.convert"
ARTIFACT_CONVERT_EXECUTOR_ID = "artifact.convert.executor"
ARTIFACT_CONVERT_TOOL_NAME = "artifact_convert"
ARTIFACT_CONVERT_OUTPUT_KIND = "artifact.conversion"

ARTIFACT_EDIT_TEXT_CAPABILITY_ID = "artifact.edit_text"
ARTIFACT_EDIT_TEXT_EXECUTOR_ID = "artifact.edit_text.executor"
ARTIFACT_EDIT_TEXT_TOOL_NAME = "artifact_edit_text"
ARTIFACT_EDIT_TEXT_OUTPUT_KIND = "artifact.text_edit"

ARTIFACT_SAVE_LOCAL_CAPABILITY_ID = "artifact.save_local"
ARTIFACT_SAVE_LOCAL_EXECUTOR_ID = "artifact.save_local.executor"
ARTIFACT_SAVE_LOCAL_TOOL_NAME = "artifact_save_local"
ARTIFACT_DELIVERY_RECEIPT_OUTPUT_KIND = "artifact.delivery_receipt"

ARTIFACT_SET_EXPORT_LOCATION_CAPABILITY_ID = "artifact.set_export_location"
ARTIFACT_SET_EXPORT_LOCATION_EXECUTOR_ID = "artifact.set_export_location.executor"
ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME = "artifact_set_export_location"
ARTIFACT_EXPORT_LOCATION_OUTPUT_KIND = "artifact.export_location"
ARTIFACT_DOMAIN_OWNER_ID = "artifacts"


@dataclass(frozen=True, slots=True)
class ArtifactCapabilityDeclarations:
    capabilities: tuple[Capability, ...]
    executors: tuple[Executor, ...]
    tool_views: tuple[ToolView, ...]


@dataclass(frozen=True, slots=True)
class ExactTabularExportResult:
    """One complete typed-source export after its selected fixed renderer."""

    format: str
    source_id: str
    source_revision: str
    sql_fingerprint: str
    resource_revisions: tuple[tuple[str, str], ...]
    columns: tuple[str, ...]
    row_count: int
    content: bytes
    sensitivity: Sensitivity

    def __post_init__(self) -> None:
        if self.format not in {"csv", "xlsx"}:
            raise ValueError("exact tabular format is invalid")
        for value, name in (
            (self.source_id, "source_id"),
            (self.source_revision, "source_revision"),
            (self.sql_fingerprint, "sql_fingerprint"),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"exact tabular {name} must be non-empty text")
        if not self.sql_fingerprint.startswith("sha256:"):
            raise ValueError("exact tabular sql_fingerprint must use sha256")
        revisions = tuple(sorted(tuple(item) for item in self.resource_revisions))
        if not revisions or len(revisions) > 64:
            raise ValueError("exact tabular resource revisions exceed their bound")
        if len({item[0] for item in revisions}) != len(revisions) or any(
            not resource_id or not revision.startswith("sha256:")
            for resource_id, revision in revisions
        ):
            raise ValueError("exact tabular resource revisions are invalid")
        columns = tuple(self.columns)
        if (
            not 1 <= len(columns) <= MAX_CSV_COLUMNS
            or len(columns) != len(set(columns))
            or any(
                not isinstance(column, str) or not column.strip() or len(column) > 256
                for column in columns
            )
        ):
            raise ValueError("exact tabular columns are invalid")
        if (
            not isinstance(self.row_count, int)
            or isinstance(self.row_count, bool)
            or not 0 <= self.row_count <= MAX_CSV_ROWS
        ):
            raise ValueError("exact tabular row_count is outside its bound")
        if (
            not isinstance(self.content, bytes)
            or not self.content
            or len(self.content) > MAX_CSV_BYTES
        ):
            raise ValueError("exact tabular content is outside its byte bound")
        if (
            not isinstance(self.sensitivity, Sensitivity)
            or self.sensitivity is Sensitivity.UNKNOWN
        ):
            raise ValueError("exact tabular sensitivity must be resolved")
        object.__setattr__(self, "resource_revisions", revisions)
        object.__setattr__(self, "columns", columns)


ExactTabularProgress = Callable[[int, int, int], None]


class ExactTabularExportBackend(Protocol):
    async def execute_exact_tabular(
        self,
        *,
        agent_id: str,
        source_id: str,
        sql: str,
        parameters: tuple[object, ...],
        format_name: str,
        parameters_sha256: str,
        created_at: datetime,
        max_rows: int,
        max_columns: int,
        max_bytes: int,
        timeout_seconds: float,
        progress: ExactTabularProgress | None = None,
    ) -> ExactTabularExportResult: ...


def resolved_exact_export_sensitivity(
    values: tuple[str | Sensitivity, ...],
) -> Sensitivity:
    """Resolve current resource labels at the exact-artifact trust boundary."""

    order = {
        Sensitivity.PUBLIC: 0,
        Sensitivity.INTERNAL: 1,
        Sensitivity.CONFIDENTIAL: 2,
        Sensitivity.RESTRICTED: 3,
        Sensitivity.UNKNOWN: 3,
    }
    resolved: list[Sensitivity] = []
    for value in values:
        try:
            resolved.append(
                value if isinstance(value, Sensitivity) else Sensitivity(value)
            )
        except ValueError:
            resolved.append(Sensitivity.RESTRICTED)
    selected = max(resolved or [Sensitivity.RESTRICTED], key=order.__getitem__)
    return Sensitivity.RESTRICTED if selected is Sensitivity.UNKNOWN else selected


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


class ResultSnapshotExecutor:
    executor_id = RESULT_SNAPSHOT_EXECUTOR_ID

    async def execute(self, request: ToolExecution) -> ToolOutput:
        filename = request.arguments.get("filename")
        result_data = request.arguments["_result_data"]
        producer_provenance = request.arguments["_producer_provenance"]
        assert filename is None or isinstance(filename, str)
        assert isinstance(result_data, Mapping)
        assert isinstance(producer_provenance, Mapping)
        snapshot = serialize_result_snapshot(result_data)
        expected_digest = request.arguments["_result_sha256"]
        if snapshot.sha256 != expected_digest:
            raise ToolOutputValidationError(
                "result snapshot content changed after evidence validation"
            )
        observed_at = request.arguments["_observed_at"]
        sensitivity = request.arguments["_result_sensitivity"]
        assert isinstance(observed_at, str)
        assert isinstance(sensitivity, str)
        binding = ArtifactResultBinding(
            capability_id=str(request.arguments["_producer_capability_id"]),
            executor_id=str(request.arguments["_producer_executor_id"]),
            call_id=str(request.arguments["call_id"]),
            capability_contract_digest=str(
                request.arguments["_capability_contract_digest"]
            ),
            output_schema_digest=str(request.arguments["_output_schema_digest"]),
            arguments_sha256=str(request.arguments["_arguments_sha256"]),
            result_sha256=snapshot.sha256,
            observed_at=datetime.fromisoformat(observed_at).astimezone(UTC),
            result_sensitivity=Sensitivity(sensitivity),
            producer_provenance=producer_provenance,
        )
        safe_filename = canonical_artifact_filename(
            filename or "result.json",
            RESULT_SNAPSHOT_MEDIA_TYPE,
            RESULT_SNAPSHOT_ALLOWED_EXTENSIONS,
        )
        draft = ArtifactDraft(
            content=snapshot.content,
            suggested_filename=safe_filename,
            media_type=RESULT_SNAPSHOT_MEDIA_TYPE,
            sensitivity=binding.result_sensitivity,
            provenance=ArtifactProvenance(
                authorship=ArtifactAuthorship.VALIDATED_TOOL_RESULT,
                evidence_call_ids=(binding.call_id,),
                result_binding=binding,
            ),
        )
        return ToolOutput(
            kind=RESULT_SNAPSHOT_OUTPUT_KIND,
            data={
                "filename": safe_filename,
                "byte_size": len(snapshot.content),
                "source_call_id": binding.call_id,
                "result_sha256": snapshot.sha256,
            },
            artifact=draft,
            sensitivity=ModelSensitivity(sensitivity),
            sensitivity_provenance={
                "authority": "validated_current_run_tool_result",
                "source_call_id": binding.call_id,
                "producer_capability_id": binding.capability_id,
                "result_sha256": binding.result_sha256,
            },
        )


class _TabularExportExecutor:
    executor_id: str

    def __init__(
        self,
        agent_id: str,
        backend: ExactTabularExportBackend,
        *,
        clock: Callable[[], datetime],
        max_seconds: float = MAX_CSV_SECONDS,
    ) -> None:
        self._agent_id = agent_id
        self._backend = backend
        self._clock = clock
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
        assert isinstance(format_name, str) and format_name in {"csv", "xlsx"}
        assert filename is None or isinstance(filename, str)
        completed = {"rows": 0, "columns": 0, "bytes": 0}

        def progress(rows: int, columns: int, byte_count: int) -> None:
            completed.update(rows=rows, columns=columns, bytes=byte_count)

        parameters_sha256 = (
            "sha256:" + sha256(canonical_json(parameters).encode("utf-8")).hexdigest()
        )
        try:
            async with asyncio.timeout(self._max_seconds):
                result = await self._backend.execute_exact_tabular(
                    agent_id=self._agent_id,
                    source_id=source_id,
                    sql=sql,
                    parameters=parameters,
                    format_name=format_name,
                    parameters_sha256=parameters_sha256,
                    created_at=self._clock(),
                    max_rows=MAX_CSV_ROWS,
                    max_columns=MAX_CSV_COLUMNS,
                    max_bytes=MAX_CSV_BYTES,
                    timeout_seconds=self._max_seconds,
                    progress=progress,
                )
        except TimeoutError as error:
            raise ArtifactError(
                "artifact_incomplete_export",
                "The exact tabular export exceeded its execution-time limit.",
                {
                    "reason": "time_limit",
                    "completed_rows": completed["rows"],
                    "completed_columns": completed["columns"],
                    "completed_bytes": completed["bytes"],
                },
            ) from error
        if result.source_id != source_id or result.format != format_name:
            raise ValueError("exact tabular backend returned different execution facts")
        media_type = "text/csv" if format_name == "csv" else XLSX_MEDIA_TYPE
        extensions = (
            CSV_ALLOWED_EXTENSIONS if format_name == "csv" else XLSX_ALLOWED_EXTENSIONS
        )
        safe_filename = canonical_artifact_filename(
            filename or f"export.{format_name}",
            media_type,
            extensions,
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
        draft = ArtifactDraft(
            content=result.content,
            suggested_filename=safe_filename,
            media_type=media_type,
            sensitivity=result.sensitivity,
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
            kind=TABULAR_EXPORT_OUTPUT_KIND,
            data={
                "format": format_name,
                "filename": safe_filename,
                "row_count": result.row_count,
                "column_count": len(result.columns),
            },
            artifact=draft,
        )


class SQLiteTabularExportExecutor(_TabularExportExecutor):
    executor_id = SQLITE_TABULAR_EXPORT_EXECUTOR_ID


class PostgreSQLTabularExportExecutor(_TabularExportExecutor):
    executor_id = POSTGRESQL_TABULAR_EXPORT_EXECUTOR_ID


class ArtifactListExecutor:
    executor_id = ARTIFACT_LIST_EXECUTOR_ID

    def __init__(self, artifacts: AgentHomeArtifactStore) -> None:
        self._artifacts = artifacts

    async def execute(self, request: ToolExecution) -> ToolOutput:
        conversation_id = request.conversation_id or request.run_id
        refs = await self._artifacts.list_refs(conversation_id=conversation_id)
        selected = tuple(reversed(refs[-MAX_MODEL_ARTIFACT_LIST_ITEMS:]))
        return ToolOutput(
            kind=ARTIFACT_LIST_OUTPUT_KIND,
            data={
                "artifacts": tuple(_model_artifact_summary(ref) for ref in selected),
                "returned_count": len(selected),
                "truncated": len(refs) > len(selected),
            },
        )


class ArtifactReadExecutor:
    executor_id = ARTIFACT_READ_EXECUTOR_ID

    def __init__(self, artifacts: AgentHomeArtifactStore) -> None:
        self._artifacts = artifacts

    async def execute(self, request: ToolExecution) -> ToolOutput:
        artifact_id = request.arguments["artifact_id"]
        assert isinstance(artifact_id, str)
        payload = await self._artifacts.read(artifact_id)
        summary = _model_artifact_summary(payload.ref)
        if payload.ref.media_type == XLSX_MEDIA_TYPE:
            data = await asyncio.to_thread(read_exact_xlsx_data, payload.content)
            rows = data.rows[:MAX_MODEL_ARTIFACT_XLSX_ROWS]
            return ToolOutput(
                kind=ARTIFACT_READ_OUTPUT_KIND,
                data={
                    "artifact": summary,
                    "representation": "xlsx_data",
                    "columns": data.columns,
                    "rows": tuple(
                        tuple(_model_cell(value) for value in row) for row in rows
                    ),
                    "total_rows": len(data.rows),
                    "returned_rows": len(rows),
                    "truncated": len(rows) < len(data.rows),
                },
            )
        if payload.ref.media_type not in TEXT_EDIT_MEDIA_TYPES:
            raise ArtifactError(
                "artifact_invalid_format",
                "This artifact format does not support a model preview.",
                {"media_type": payload.ref.media_type},
            )
        try:
            payload.content.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ArtifactError(
                "artifact_invalid_format",
                "This artifact is not valid UTF-8 text.",
                {"media_type": payload.ref.media_type},
            ) from error
        preview_bytes = payload.content[:MAX_MODEL_ARTIFACT_TEXT_BYTES]
        try:
            preview = preview_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            preview_bytes = preview_bytes[: error.start]
            preview = preview_bytes.decode("utf-8")
        return ToolOutput(
            kind=ARTIFACT_READ_OUTPUT_KIND,
            data={
                "artifact": summary,
                "representation": "utf8_text",
                "text": preview,
                "total_utf8_bytes": len(payload.content),
                "returned_utf8_bytes": len(preview_bytes),
                "truncated": len(preview_bytes) < len(payload.content),
            },
        )


class ArtifactConvertExecutor:
    executor_id = ARTIFACT_CONVERT_EXECUTOR_ID

    def __init__(self, artifacts: AgentHomeArtifactStore) -> None:
        self._artifacts = artifacts

    async def execute(self, request: ToolExecution) -> ToolOutput:
        artifact_id = request.arguments["artifact_id"]
        format_name = request.arguments["format"]
        filename = request.arguments.get("filename")
        assert isinstance(artifact_id, str)
        assert format_name == "csv"
        assert filename is None or isinstance(filename, str)
        payload = await _read_conversation_artifact(
            self._artifacts,
            request.conversation_id or request.run_id,
            artifact_id,
        )
        if (
            payload.ref.media_type != XLSX_MEDIA_TYPE
            or payload.ref.capability_id
            not in {
                SQLITE_TABULAR_EXPORT_CAPABILITY_ID,
                POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
            }
            or payload.ref.provenance.authorship
            is not ArtifactAuthorship.EXACT_SOURCE_DATA
        ):
            raise ArtifactError(
                "artifact_invalid_format",
                "Only a Daita-generated exact XLSX artifact can be converted to CSV.",
                {
                    "media_type": payload.ref.media_type,
                    "allowed_extensions": (".xlsx",),
                },
            )
        try:
            async with asyncio.timeout(MAX_XLSX_SECONDS):
                data = await asyncio.to_thread(read_exact_xlsx_data, payload.content)
                content = await asyncio.to_thread(
                    render_exact_csv,
                    data.columns,
                    data.rows,
                )
        except TimeoutError as error:
            raise ArtifactError(
                "artifact_incomplete_export",
                "The artifact conversion exceeded its execution-time limit.",
                {
                    "reason": "time_limit",
                    "completed_rows": 0,
                    "completed_columns": 0,
                    "completed_bytes": 0,
                },
            ) from error
        suggested = filename or payload.ref.filename.rsplit(".", 1)[0] + ".csv"
        safe_filename = canonical_artifact_filename(
            suggested,
            "text/csv",
            CSV_ALLOWED_EXTENSIONS,
        )
        source_provenance = payload.ref.provenance
        draft = ArtifactDraft(
            content=content,
            suggested_filename=safe_filename,
            media_type="text/csv",
            sensitivity=payload.ref.sensitivity,
            provenance=ArtifactProvenance(
                authorship=ArtifactAuthorship.EXACT_SOURCE_DATA,
                derived_from_artifact_id=payload.ref.artifact_id,
                resource_bindings=source_provenance.resource_bindings,
                sql_fingerprint=source_provenance.sql_fingerprint,
                parameters_sha256=source_provenance.parameters_sha256,
                columns=data.columns,
                row_count=len(data.rows),
            ),
        )
        return ToolOutput(
            kind=ARTIFACT_CONVERT_OUTPUT_KIND,
            data={
                "source_artifact_id": payload.ref.artifact_id,
                "format": "csv",
                "filename": safe_filename,
                "row_count": len(data.rows),
                "column_count": len(data.columns),
            },
            artifact=draft,
        )


class ArtifactEditTextExecutor:
    executor_id = ARTIFACT_EDIT_TEXT_EXECUTOR_ID

    def __init__(self, workspace: LocalWorkspaceBackend) -> None:
        self._workspace = workspace

    async def execute(self, request: ToolExecution) -> ToolOutput:
        token = request.arguments["binding"]
        replacements = request.arguments["replacements"]
        assert isinstance(token, str)
        assert isinstance(replacements, tuple)
        observation = await self._workspace.observe_bound_text(
            run_id=request.run_id,
            token=token,
        )
        transformed = await asyncio.to_thread(
            apply_bounded_text_edits,
            source=observation.content,
            relative_path=observation.binding.relative_path,
            replacements=tuple(
                item for item in replacements if isinstance(item, Mapping)
            ),
        )
        summary = transformed.summary
        binding = ArtifactLocalFileBinding(
            workspace_id=observation.binding.workspace_id,
            relative_path=observation.binding.relative_path,
            original_physical_revision=observation.binding.physical_revision,
            observed_content_sha256=observation.content_sha256,
            source_byte_size=len(observation.content),
            change_summary=summary,
        )
        filename = observation.binding.relative_path.rsplit("/", 1)[-1]
        sensitivity = Sensitivity(self._workspace.sensitivity.value)
        draft = ArtifactDraft(
            content=transformed.content,
            suggested_filename=filename,
            media_type=transformed.media_type,
            sensitivity=sensitivity,
            provenance=ArtifactProvenance(
                authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
                local_file_binding=binding,
            ),
        )
        return ToolOutput(
            kind=ARTIFACT_EDIT_TEXT_OUTPUT_KIND,
            data={
                "relative_path": binding.relative_path,
                "filename": filename,
                "original_physical_revision": binding.original_physical_revision,
                "observed_content_sha256": binding.observed_content_sha256,
                "source_byte_size": binding.source_byte_size,
                "result_byte_size": len(transformed.content),
                "change_summary": artifact_text_change_summary_to_mapping(summary),
            },
            artifact=draft,
            sensitivity=ModelSensitivity(self._workspace.sensitivity.value),
            sensitivity_provenance={
                "authority": "local_workspace_binding",
                "workspace_id": self._workspace.workspace_id,
                "relative_paths": (binding.relative_path,),
                "physical_revisions": (binding.original_physical_revision,),
            },
        )


async def _read_conversation_artifact(
    artifacts: AgentHomeArtifactStore,
    conversation_id: str,
    artifact_id: str,
) -> ArtifactPayload:
    ref = next(
        (
            item
            for item in await artifacts.list_refs(conversation_id=conversation_id)
            if item.artifact_id == artifact_id
        ),
        None,
    )
    if ref is None:
        raise ArtifactError(
            "artifact_missing",
            "The requested artifact is not available in the current conversation.",
            {"artifact_id": artifact_id},
        )
    return await artifacts.read_ref(ref)


def _model_artifact_summary(ref: ArtifactRef) -> dict[str, object]:
    summary: dict[str, object] = {
        "artifact_id": ref.artifact_id,
        "filename": ref.filename,
        "media_type": ref.media_type,
        "byte_size": ref.byte_size,
        "sha256": ref.sha256,
        "created_at": ref.created_at.isoformat().replace("+00:00", "Z"),
    }
    if ref.provenance.derived_from_artifact_id is not None:
        summary["derived_from_artifact_id"] = ref.provenance.derived_from_artifact_id
    return summary


def _model_cell(value: object) -> object:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


class ArtifactSaveLocalExecutor:
    executor_id = ARTIFACT_SAVE_LOCAL_EXECUTOR_ID

    def __init__(self, delivery: LocalArtifactDelivery) -> None:
        self._delivery = delivery

    async def preflight(self, request: ToolExecution) -> FrozenJsonObject:
        artifact_id = request.arguments["artifact_id"]
        mode = request.arguments["mode"]
        destination_id = request.arguments.get("destination_id")
        filename = request.arguments.get("filename")
        assert isinstance(artifact_id, str)
        assert isinstance(mode, str)
        assert destination_id is None or isinstance(destination_id, str)
        assert filename is None or isinstance(filename, str)
        return await self._delivery.preflight_save(
            run_id=request.run_id,
            artifact_id=artifact_id,
            mode=mode,
            destination_id=destination_id,
            filename=filename,
        )

    async def execute(self, request: ToolExecution) -> ToolOutput:
        artifact_id = request.arguments["artifact_id"]
        mode = request.arguments["mode"]
        destination_id = request.arguments.get("destination_id")
        filename = request.arguments.get("filename")
        assert isinstance(artifact_id, str)
        assert isinstance(mode, str)
        assert destination_id is None or isinstance(destination_id, str)
        assert filename is None or isinstance(filename, str)
        receipt = await self._delivery.save_committed(
            run_id=request.run_id,
            artifact_id=artifact_id,
            mode=mode,
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


def artifact_declarations(
    delivery: LocalArtifactDelivery | None,
    artifacts: AgentHomeArtifactStore,
    *,
    workspace: LocalWorkspaceBackend | None,
    agent_id: str,
    sqlite_backend: ExactTabularExportBackend,
    postgresql_backend: ExactTabularExportBackend,
    clock: Callable[[], datetime],
) -> ArtifactCapabilityDeclarations:
    include_local_delivery = delivery is not None
    declarations = artifact_capability_declarations(
        include_local_delivery=include_local_delivery,
        include_local_edit=workspace is not None,
    )
    delivery_executors: tuple[Executor, ...] = ()
    if delivery is not None:
        delivery_executors = (
            ArtifactSaveLocalExecutor(delivery),
            ArtifactSetExportLocationExecutor(delivery),
        )
    return ArtifactCapabilityDeclarations(
        capabilities=declarations.capabilities,
        executors=(
            DocumentArtifactExecutor(),
            ResultSnapshotExecutor(),
            SQLiteTabularExportExecutor(agent_id, sqlite_backend, clock=clock),
            PostgreSQLTabularExportExecutor(agent_id, postgresql_backend, clock=clock),
            ArtifactListExecutor(artifacts),
            ArtifactReadExecutor(artifacts),
            ArtifactConvertExecutor(artifacts),
            *((ArtifactEditTextExecutor(workspace),) if workspace is not None else ()),
            *delivery_executors,
        ),
        tool_views=declarations.tool_views,
    )


def artifact_capability_declarations(
    *,
    include_local_delivery: bool = True,
    include_local_edit: bool = True,
) -> CapabilityDeclarations:
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
        automation_eligibility=AutomationEligibility.SCHEDULED_DIRECT,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({"text/markdown", "text/plain"}),
            allowed_authorships=frozenset({ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS}),
            allowed_extensions=DOCUMENT_ALLOWED_EXTENSIONS,
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=MAX_DOCUMENT_BYTES,
            max_total_bytes_per_call=MAX_DOCUMENT_BYTES,
        ),
    )
    result_snapshot = Capability(
        id=RESULT_SNAPSHOT_CAPABILITY_ID,
        description=(
            "Create one canonical JSON artifact from the exact validated output "
            "data of one earlier successful tool call in the current run."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "call_id": {"type": "string", "minLength": 1, "maxLength": 256},
                "filename": {"type": "string", "minLength": 1, "maxLength": 120},
            },
            "required": ["call_id"],
            "additionalProperties": False,
        },
        output_kind=RESULT_SNAPSHOT_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "filename": {"type": "string"},
                "byte_size": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESULT_SNAPSHOT_BYTES,
                },
                "source_call_id": {"type": "string"},
                "result_sha256": {
                    "type": "string",
                    "pattern": "^sha256:[0-9a-f]{64}$",
                },
            },
            "required": [
                "filename",
                "byte_size",
                "source_call_id",
                "result_sha256",
            ],
            "additionalProperties": False,
        },
        executor_id=RESULT_SNAPSHOT_EXECUTOR_ID,
        automation_eligibility=AutomationEligibility.SCHEDULED_DIRECT,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({RESULT_SNAPSHOT_MEDIA_TYPE}),
            allowed_authorships=frozenset({ArtifactAuthorship.VALIDATED_TOOL_RESULT}),
            allowed_extensions=RESULT_SNAPSHOT_ALLOWED_EXTENSIONS,
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=MAX_RESULT_SNAPSHOT_BYTES,
            max_total_bytes_per_call=MAX_RESULT_SNAPSHOT_BYTES,
        ),
    )
    sqlite_tabular = _tabular_export_capability(
        capability_id=SQLITE_TABULAR_EXPORT_CAPABILITY_ID,
        executor_id=SQLITE_TABULAR_EXPORT_EXECUTOR_ID,
        adapter_name="SQLite",
    )
    postgresql_tabular = _tabular_export_capability(
        capability_id=POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
        executor_id=POSTGRESQL_TABULAR_EXPORT_EXECUTOR_ID,
        adapter_name="PostgreSQL",
    )
    artifact_list = Capability(
        id=ARTIFACT_LIST_CAPABILITY_ID,
        description=(
            "List a bounded newest-first set of safe artifact metadata from only "
            "the current conversation."
        ),
        input_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        output_kind=ARTIFACT_LIST_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "artifacts": {
                    "type": "array",
                    "items": _artifact_summary_schema(),
                    "maxItems": MAX_MODEL_ARTIFACT_LIST_ITEMS,
                },
                "returned_count": {"type": "integer", "minimum": 0},
                "truncated": {"type": "boolean"},
            },
            "required": ["artifacts", "returned_count", "truncated"],
            "additionalProperties": False,
        },
        executor_id=ARTIFACT_LIST_EXECUTOR_ID,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    artifact_read = Capability(
        id=ARTIFACT_READ_CAPABILITY_ID,
        description=(
            "Read one bounded truthful preview of an exact known artifact owned by "
            "the current agent, including the fixed Data sheet of a Daita-generated "
            "XLSX workbook."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "pattern": "^artifact-[0-9a-f]{32}$",
                }
            },
            "required": ["artifact_id"],
            "additionalProperties": False,
        },
        output_kind=ARTIFACT_READ_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "artifact": _artifact_summary_schema(),
                "representation": {
                    "type": "string",
                    "enum": ["utf8_text", "xlsx_data"],
                },
                "text": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "rows": {"type": "array", "items": {"type": "array"}},
                "total_rows": {"type": "integer", "minimum": 0},
                "returned_rows": {"type": "integer", "minimum": 0},
                "total_utf8_bytes": {"type": "integer", "minimum": 0},
                "returned_utf8_bytes": {"type": "integer", "minimum": 0},
                "truncated": {"type": "boolean"},
            },
            "required": ["artifact", "representation", "truncated"],
            "additionalProperties": False,
        },
        executor_id=ARTIFACT_READ_EXECUTOR_ID,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    artifact_convert = Capability(
        id=ARTIFACT_CONVERT_CAPABILITY_ID,
        description=(
            "Convert one current-conversation Daita-generated XLSX snapshot to "
            "exact CSV without rerunning its source or routing bytes through the model."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "pattern": "^artifact-[0-9a-f]{32}$",
                },
                "format": {"type": "string", "enum": ["csv"]},
                "filename": {"type": "string", "minLength": 1, "maxLength": 120},
            },
            "required": ["artifact_id", "format"],
            "additionalProperties": False,
        },
        output_kind=ARTIFACT_CONVERT_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "source_artifact_id": {"type": "string"},
                "format": {"type": "string", "enum": ["csv"]},
                "filename": {"type": "string"},
                "row_count": {"type": "integer", "minimum": 0},
                "column_count": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_CSV_COLUMNS,
                },
            },
            "required": [
                "source_artifact_id",
                "format",
                "filename",
                "row_count",
                "column_count",
            ],
            "additionalProperties": False,
        },
        executor_id=ARTIFACT_CONVERT_EXECUTOR_ID,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({"text/csv"}),
            allowed_authorships=frozenset({ArtifactAuthorship.EXACT_SOURCE_DATA}),
            allowed_extensions=CSV_ALLOWED_EXTENSIONS,
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=MAX_CSV_BYTES,
            max_total_bytes_per_call=MAX_CSV_BYTES,
        ),
    )
    artifact_edit_text = Capability(
        id=ARTIFACT_EDIT_TEXT_CAPABILITY_ID,
        description=(
            "Apply one bounded ordered exact UTF-8 text replacement family to an "
            "authenticated current-run file_read binding and commit the complete "
            "result as an internal artifact without changing the workspace file. "
            "Copy file_read data.binding verbatim; it is opaque and must never be "
            "decoded, normalized, or reconstructed."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "binding": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 8 * 1024,
                    "description": (
                        "The exact opaque file_read data.binding string copied "
                        "verbatim without decoding, normalization, or reconstruction."
                    ),
                },
                "replacements": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_TEXT_EDIT_OPERATIONS,
                    "items": {
                        "type": "object",
                        "properties": {
                            "old_text": {
                                "type": "string",
                                "minLength": 1,
                                "maxLength": MAX_TEXT_EDIT_ANCHOR_BYTES,
                            },
                            "new_text": {
                                "type": "string",
                                "maxLength": MAX_TEXT_EDIT_REPLACEMENT_BYTES,
                            },
                            "expected_occurrences": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": MAX_TEXT_EDIT_OCCURRENCES,
                            },
                        },
                        "required": [
                            "old_text",
                            "new_text",
                            "expected_occurrences",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["binding", "replacements"],
            "additionalProperties": False,
        },
        output_kind=ARTIFACT_EDIT_TEXT_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "relative_path": {"type": "string"},
                "filename": {"type": "string"},
                "original_physical_revision": {"type": "string"},
                "observed_content_sha256": {"type": "string"},
                "source_byte_size": {"type": "integer", "minimum": 0},
                "result_byte_size": {"type": "integer", "minimum": 0},
                "change_summary": _change_summary_schema(),
            },
            "required": [
                "relative_path",
                "filename",
                "original_physical_revision",
                "observed_content_sha256",
                "source_byte_size",
                "result_byte_size",
                "change_summary",
            ],
            "additionalProperties": False,
        },
        executor_id=ARTIFACT_EDIT_TEXT_EXECUTOR_ID,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=TEXT_EDIT_MEDIA_TYPES,
            allowed_authorships=frozenset({ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS}),
            allowed_extensions=TEXT_EDIT_ALLOWED_EXTENSIONS,
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=MAX_TEXT_EDIT_BYTES,
            max_total_bytes_per_call=MAX_TEXT_EDIT_BYTES,
        ),
        access_mode=AccessMode.READ,
    )
    save = Capability(
        id=ARTIFACT_SAVE_LOCAL_CAPABILITY_ID,
        description=(
            "Publish one committed artifact either as a collision-safe new file in "
            "an authorized destination or as an atomic replacement of only the exact "
            "unchanged workspace file bound into a text-edit artifact."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "pattern": "^artifact-[0-9a-f]{32}$",
                },
                "mode": {
                    "type": "string",
                    "enum": ["create_new", "replace_bound_file"],
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
            "required": ["artifact_id", "mode"],
            "additionalProperties": False,
        },
        output_kind=ARTIFACT_DELIVERY_RECEIPT_OUTPUT_KIND,
        output_schema=_receipt_schema(),
        executor_id=ARTIFACT_SAVE_LOCAL_EXECUTOR_ID,
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.CHANGE_INFRASTRUCTURE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
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
        access_mode=AccessMode.NONE,
        operational_effect=OperationalEffect.CHANGE_INFRASTRUCTURE,
        automation_eligibility=AutomationEligibility.INTERACTIVE_ONLY,
    )
    capabilities = (
        document,
        result_snapshot,
        sqlite_tabular,
        postgresql_tabular,
        artifact_list,
        artifact_read,
        artifact_convert,
        *((artifact_edit_text,) if include_local_edit else ()),
        *((save, set_location) if include_local_delivery else ()),
    )
    views = (
        ToolView(
            name=DOCUMENT_CREATE_TOOL_NAME,
            capability_id=document.id,
            description=document.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ARTIFACTS,
                load_mode=ToolLoadMode.ON_DEMAND,
                text_trust=ToolTextTrust.CODE,
                summary="Create a bounded Markdown or text document artifact.",
                when_to_use="Use when the requested deliverable is a document.",
                keywords=("artifact", "document", "markdown", "text"),
            ),
        ),
        ToolView(
            name=RESULT_SNAPSHOT_TOOL_NAME,
            capability_id=result_snapshot.id,
            description=result_snapshot.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ARTIFACTS,
                load_mode=ToolLoadMode.ON_DEMAND,
                text_trust=ToolTextTrust.CODE,
                summary="Snapshot one earlier structured result as canonical JSON.",
                when_to_use=(
                    "Use after a successful current-run tool call when its complete "
                    "validated structured result should become an artifact."
                ),
                keywords=("artifact", "snapshot", "result", "json"),
            ),
        ),
        ToolView(
            name=SQLITE_TABULAR_EXPORT_TOOL_NAME,
            capability_id=sqlite_tabular.id,
            description=sqlite_tabular.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ARTIFACTS,
                load_mode=ToolLoadMode.ON_DEMAND,
                text_trust=ToolTextTrust.CODE,
                summary="Export an exact SQLite query result as CSV or XLSX.",
                when_to_use="Use for a downloadable tabular SQLite result.",
                keywords=("artifact", "export", "sqlite", "csv", "xlsx"),
            ),
        ),
        ToolView(
            name=POSTGRESQL_TABULAR_EXPORT_TOOL_NAME,
            capability_id=postgresql_tabular.id,
            description=postgresql_tabular.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ARTIFACTS,
                load_mode=ToolLoadMode.ON_DEMAND,
                text_trust=ToolTextTrust.CODE,
                summary="Export an exact PostgreSQL query result as CSV or XLSX.",
                when_to_use="Use for a downloadable tabular PostgreSQL result.",
                keywords=("artifact", "export", "postgresql", "csv", "xlsx"),
            ),
        ),
        ToolView(
            name=ARTIFACT_LIST_TOOL_NAME,
            capability_id=artifact_list.id,
            description=artifact_list.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ARTIFACTS,
                load_mode=ToolLoadMode.PINNED,
                text_trust=ToolTextTrust.CODE,
                summary="List bounded safe metadata for current-conversation artifacts.",
                when_to_use="Use to identify an artifact created earlier in the conversation.",
                keywords=("artifact", "list", "conversation", "file"),
            ),
        ),
        ToolView(
            name=ARTIFACT_READ_TOOL_NAME,
            capability_id=artifact_read.id,
            description=artifact_read.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ARTIFACTS,
                load_mode=ToolLoadMode.PINNED,
                text_trust=ToolTextTrust.CODE,
                summary="Read a bounded safe preview of one exact agent-owned artifact.",
                when_to_use=(
                    "Use with an exact known artifact ID, including one returned by "
                    "job_read_results."
                ),
                keywords=("artifact", "read", "preview", "file"),
            ),
        ),
        ToolView(
            name=ARTIFACT_CONVERT_TOOL_NAME,
            capability_id=artifact_convert.id,
            description=artifact_convert.description,
            presentation=ToolPresentation(
                toolbox_id=ToolboxId.ARTIFACTS,
                load_mode=ToolLoadMode.ON_DEMAND,
                text_trust=ToolTextTrust.CODE,
                summary="Convert a verified Daita XLSX Data snapshot to CSV.",
                when_to_use="Use for exact supported conversion of an existing artifact.",
                keywords=("artifact", "convert", "xlsx", "csv"),
            ),
        ),
        *(
            (
                ToolView(
                    name=ARTIFACT_EDIT_TEXT_TOOL_NAME,
                    capability_id=artifact_edit_text.id,
                    description=artifact_edit_text.description,
                    presentation=ToolPresentation(
                        toolbox_id=ToolboxId.ARTIFACTS,
                        load_mode=ToolLoadMode.ON_DEMAND,
                        text_trust=ToolTextTrust.CODE,
                        summary="Prepare a bounded exact UTF-8 workspace-file edit artifact.",
                        when_to_use=(
                            "Use only with an authenticated binding returned by file_read; "
                            "then use artifact_save_local in replace_bound_file mode."
                        ),
                        keywords=("artifact", "edit", "text", "replace", "workspace"),
                    ),
                ),
            )
            if include_local_edit
            else ()
        ),
        *(
            (
                ToolView(
                    name=ARTIFACT_SAVE_LOCAL_TOOL_NAME,
                    capability_id=save.id,
                    description=save.description,
                    presentation=ToolPresentation(
                        toolbox_id=ToolboxId.ARTIFACTS,
                        load_mode=ToolLoadMode.ON_DEMAND,
                        text_trust=ToolTextTrust.CODE,
                        summary="Deliver one committed artifact to an authorized local destination.",
                        when_to_use="Use after artifact creation when local delivery is required.",
                        keywords=("artifact", "save", "deliver", "local"),
                    ),
                ),
                ToolView(
                    name=ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME,
                    capability_id=set_location.id,
                    description=set_location.description,
                    presentation=ToolPresentation(
                        toolbox_id=ToolboxId.ARTIFACTS,
                        load_mode=ToolLoadMode.ON_DEMAND,
                        text_trust=ToolTextTrust.CODE,
                        summary="Set an authorized destination as the future export default.",
                        when_to_use="Use only when the user explicitly changes the persistent default.",
                        keywords=("artifact", "destination", "export", "default"),
                    ),
                ),
            )
            if include_local_delivery
            else ()
        ),
    )
    return CapabilityDeclarations(
        domain_owner_id="artifacts",
        capabilities=capabilities,
        executor_ids=tuple(item.executor_id for item in capabilities),
        tool_views=views,
    )


_EXACT_TABULAR_CAPABILITIES = frozenset(
    {SQLITE_TABULAR_EXPORT_CAPABILITY_ID, POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID}
)
_CONVERSATION_ARTIFACT_CAPABILITIES = frozenset(
    {
        ARTIFACT_LIST_CAPABILITY_ID,
        ARTIFACT_CONVERT_CAPABILITY_ID,
    }
)
_ARTIFACT_PRODUCER_CAPABILITIES = frozenset(
    {
        DOCUMENT_CREATE_CAPABILITY_ID,
        RESULT_SNAPSHOT_CAPABILITY_ID,
        SQLITE_TABULAR_EXPORT_CAPABILITY_ID,
        POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
        ARTIFACT_CONVERT_CAPABILITY_ID,
        ARTIFACT_EDIT_TEXT_CAPABILITY_ID,
    }
)
_ARTIFACT_ADAPTER_CAPABILITIES = {
    "sqlite": frozenset({SQLITE_TABULAR_EXPORT_CAPABILITY_ID}),
    "postgresql": frozenset({POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID}),
}
_BASE_ARTIFACT_CAPABILITIES = frozenset(
    {
        DOCUMENT_CREATE_CAPABILITY_ID,
        RESULT_SNAPSHOT_CAPABILITY_ID,
        SQLITE_TABULAR_EXPORT_CAPABILITY_ID,
        POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID,
        ARTIFACT_LIST_CAPABILITY_ID,
        ARTIFACT_READ_CAPABILITY_ID,
        ARTIFACT_CONVERT_CAPABILITY_ID,
    }
)
_LOCAL_DELIVERY_CAPABILITIES = frozenset(
    {ARTIFACT_SAVE_LOCAL_CAPABILITY_ID, ARTIFACT_SET_EXPORT_LOCATION_CAPABILITY_ID}
)
LOCAL_ARTIFACT_EDIT_CAPABILITY_IDS = frozenset({ARTIFACT_EDIT_TEXT_CAPABILITY_ID})
LOCAL_ARTIFACT_EDIT_EXECUTOR_IDS = frozenset({ARTIFACT_EDIT_TEXT_EXECUTOR_ID})


class ArtifactDomainCatalog(Protocol):
    async def source_routing_facts(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[Mapping[str, object], ...]: ...

    async def source_adapter_id(self, agent_id: str, source_id: str) -> str | None: ...

    async def resource_schemas(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[ResourceSchema, ...]: ...

    async def readable_resource_ids(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> frozenset[str]: ...

    async def resource_identity(
        self,
        agent_id: str,
        resource_id: str,
    ) -> tuple[str, str, str] | None: ...

    async def is_current_tabular_file(
        self,
        agent_id: str,
        source_id: str,
        resource_id: str,
    ) -> bool: ...

    async def admitted_model_sensitivity(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> ModelSensitivity | None: ...


class ArtifactTranscriptReader(Protocol):
    async def load(self, run_id: str) -> Transcript: ...


class ArtifactCapabilityDomain:
    """Own artifact availability, provenance, conversion, and delivery rules."""

    domain_owner_id = ARTIFACT_DOMAIN_OWNER_ID

    def __init__(
        self,
        declarations: CapabilityDeclarations,
        catalog: ArtifactDomainCatalog,
        transcripts: ArtifactTranscriptReader,
        artifacts: AgentHomeArtifactStore,
        delivery: LocalArtifactDelivery | None,
        workspace: LocalWorkspaceBackend | None,
        learning: LearningCandidateGuard,
        *,
        clock: Callable[[], datetime],
        files_only_run_ids: set[str] | None = None,
    ) -> None:
        if declarations.domain_owner_id != self.domain_owner_id:
            raise ValueError("artifact declarations have the wrong domain owner")
        declared_ids = {item.id for item in declarations.capabilities}
        expected_ids = _BASE_ARTIFACT_CAPABILITIES
        if delivery is not None:
            expected_ids |= _LOCAL_DELIVERY_CAPABILITIES
        if workspace is not None:
            expected_ids |= LOCAL_ARTIFACT_EDIT_CAPABILITY_IDS
        if declared_ids != expected_ids:
            raise ValueError("artifact domain requires its exact capabilities")
        if bool(declared_ids & _LOCAL_DELIVERY_CAPABILITIES) != (delivery is not None):
            raise ValueError("artifact local delivery composition is inconsistent")
        if bool(declared_ids & LOCAL_ARTIFACT_EDIT_CAPABILITY_IDS) != (
            workspace is not None
        ):
            raise ValueError("artifact local edit composition is inconsistent")
        self._declarations = declarations
        self._catalog = catalog
        self._transcripts = transcripts
        self._artifacts = artifacts
        self._delivery = delivery
        self._workspace = workspace
        self._learning = learning
        self._clock = clock
        self._registry: CapabilityRegistry | None = None
        self._files_only_run_ids = (
            files_only_run_ids if files_only_run_ids is not None else set()
        )
        self._views = tuple(declarations.tool_views)
        self._capabilities = {item.id: item for item in declarations.capabilities}

    @property
    def declarations(self) -> CapabilityDeclarations:
        return self._declarations

    def bind_capability_registry(self, registry: CapabilityRegistry) -> None:
        """Bind the one immutable registry used to revalidate snapshot evidence."""

        if not isinstance(registry, CapabilityRegistry):
            raise TypeError("artifact capability registry is invalid")
        registry.validate_declarations(self._declarations)
        if self._registry is not None and self._registry is not registry:
            raise ValueError("artifact capability registry is already bound")
        self._registry = registry

    async def project(self, run: RunInput) -> tuple[str, ...]:
        facts = (
            ()
            if run.id in self._files_only_run_ids
            else await self._catalog.source_routing_facts(
                run.agent_id,
                (() if run.source_id is None else (run.source_id,)),
            )
        )
        adapters = {
            adapter_id
            for fact in facts
            if isinstance((adapter_id := fact.get("adapter_id")), str)
        }
        all_refs = await self._artifacts.list_refs()
        conversation_id = run.conversation_id or run.id
        refs = tuple(
            item for item in all_refs if item.conversation_id == conversation_id
        )
        has_current = any(item.run_id == run.id for item in refs)
        has_prior = any(item.run_id != run.id for item in refs)
        candidates: list[ToolView] = []
        for view in self._views:
            capability = self._capabilities[view.capability_id]
            if not self._learning.allows(
                run.id,
                view.name,
                effectful=capability.operational_effect is not OperationalEffect.NONE,
            ):
                continue
            required_adapter = next(
                (
                    adapter_id
                    for adapter_id, ids in _ARTIFACT_ADAPTER_CAPABILITIES.items()
                    if capability.id in ids
                ),
                None,
            )
            if required_adapter is not None and required_adapter not in adapters:
                continue
            candidates.append(view)
        may_have_current_artifact = (
            has_current
            or has_prior
            or any(
                self._capabilities[view.capability_id].id
                in _ARTIFACT_PRODUCER_CAPABILITIES
                for view in candidates
            )
        )
        may_have_agent_artifact = bool(all_refs) or may_have_current_artifact
        names: list[str] = []
        for view in candidates:
            capability = self._capabilities[view.capability_id]
            if capability.id in LOCAL_ARTIFACT_EDIT_CAPABILITY_IDS and (
                self._workspace is None
                or run.origin is not RunOrigin.USER
                or run.execution_scope is not None
            ):
                continue
            if capability.id in _LOCAL_DELIVERY_CAPABILITIES and (
                run.origin is not RunOrigin.USER or run.execution_scope is not None
            ):
                continue
            if (
                capability.id in _CONVERSATION_ARTIFACT_CAPABILITIES
                and not may_have_current_artifact
            ):
                continue
            if (
                capability.id == ARTIFACT_READ_CAPABILITY_ID
                and not may_have_agent_artifact
            ):
                continue
            if (
                capability.id == ARTIFACT_SAVE_LOCAL_CAPABILITY_ID
                and not may_have_agent_artifact
            ):
                continue
            names.append(view.name)
        return tuple(names)

    def normalize_arguments(
        self,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> Mapping[str, object]:
        del capability
        return arguments

    async def prepare_call(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        if capability.id == RESULT_SNAPSHOT_CAPABILITY_ID:
            return await self._prepare_result_snapshot(
                run,
                call,
                arguments,
                request_sensitivity=request_sensitivity,
            )
        del request_sensitivity
        if capability.id == ARTIFACT_EDIT_TEXT_CAPABILITY_ID:
            if (
                self._workspace is None
                or run.origin is not RunOrigin.USER
                or run.execution_scope is not None
            ):
                raise CapabilityInputError(
                    "workspace_unavailable",
                    "Workspace text editing is unavailable for this run.",
                )
            token = arguments.get("binding")
            if not isinstance(token, str):
                raise CapabilityInputError(
                    "artifact_edit_binding_invalid",
                    "Text editing requires an authenticated current-run file binding.",
                )
            try:
                binding = self._workspace.authenticate_file_binding(
                    run_id=run.id,
                    token=token,
                )
                text_edit_media_type(binding.relative_path)
            except LocalWorkspaceError as error:
                raise CapabilityInputError(
                    "artifact_edit_binding_invalid",
                    "The file binding is invalid or no longer available.",
                ) from error
            except ArtifactError as error:
                raise CapabilityInputError(
                    error.code, error.message, error.details
                ) from error
            return arguments
        if capability.operational_effect is not OperationalEffect.NONE:
            self._learning.validate_effect(run.id, call)
        supplied_source_id = arguments.get("source_id")
        if (
            run.source_id is not None
            and supplied_source_id is not None
            and supplied_source_id != run.source_id
        ):
            raise CapabilityInputError(
                "source_scope_violation",
                "This run can only access the source selected by the user.",
                {
                    "selected_source_id": run.source_id,
                    "requested_source_id": supplied_source_id,
                },
            )
        if capability.id in {
            ARTIFACT_READ_CAPABILITY_ID,
            ARTIFACT_CONVERT_CAPABILITY_ID,
        }:
            artifact_id = arguments.get("artifact_id")
            ref = None
            if isinstance(artifact_id, str):
                ref = (
                    await self._artifacts.find_ref(artifact_id)
                    if capability.id == ARTIFACT_READ_CAPABILITY_ID
                    else await self._current_ref(run, artifact_id)
                )
            if ref is None:
                raise CapabilityInputError(
                    "artifact_missing",
                    (
                        "The requested artifact is not owned by this agent."
                        if capability.id == ARTIFACT_READ_CAPABILITY_ID
                        else "The requested artifact is not available in the current conversation."
                    ),
                    {"artifact_id": artifact_id},
                )
            if capability.id == ARTIFACT_READ_CAPABILITY_ID and ref.media_type not in {
                *TEXT_EDIT_MEDIA_TYPES,
                XLSX_MEDIA_TYPE,
            }:
                raise CapabilityInputError(
                    "artifact_invalid_format",
                    "This artifact format does not support a model preview.",
                    {"media_type": ref.media_type},
                )
            if capability.id == ARTIFACT_CONVERT_CAPABILITY_ID and (
                ref.capability_id not in _EXACT_TABULAR_CAPABILITIES
                or ref.media_type != XLSX_MEDIA_TYPE
                or ref.provenance.authorship is not ArtifactAuthorship.EXACT_SOURCE_DATA
            ):
                raise CapabilityInputError(
                    "artifact_invalid_format",
                    "Only a Daita-generated exact XLSX artifact can be converted "
                    "to CSV.",
                    {"media_type": ref.media_type, "allowed_extensions": (".xlsx",)},
                )
        if capability.id == ARTIFACT_SAVE_LOCAL_CAPABILITY_ID:
            mode = arguments.get("mode")
            names = set(arguments)
            if mode == "create_new":
                if "destination_id" not in names or names - {
                    "artifact_id",
                    "mode",
                    "destination_id",
                    "filename",
                }:
                    raise CapabilityInputError(
                        "artifact_delivery_invalid",
                        "create_new requires one authorized destination and optional filename.",
                    )
            elif mode == "replace_bound_file":
                if names != {"artifact_id", "mode"}:
                    raise CapabilityInputError(
                        "artifact_edit_binding_invalid",
                        "replace_bound_file accepts only the committed artifact ID.",
                    )
                artifact_id = arguments.get("artifact_id")
                ref = (
                    await self._artifacts.find_ref(artifact_id)
                    if isinstance(artifact_id, str)
                    else None
                )
                if (
                    ref is None
                    or ref.capability_id != ARTIFACT_EDIT_TEXT_CAPABILITY_ID
                    or ref.provenance.local_file_binding is None
                ):
                    raise CapabilityInputError(
                        "artifact_edit_binding_invalid",
                        "replace_bound_file requires one committed bound edit artifact.",
                    )
            else:
                raise CapabilityInputError(
                    "artifact_delivery_invalid", "Artifact delivery mode is invalid."
                )
        if capability.id in _EXACT_TABULAR_CAPABILITIES:
            await self._validate_sql(run, capability, arguments)
        return arguments

    async def _prepare_result_snapshot(
        self,
        run: RunInput,
        call: ToolCall,
        arguments: FrozenJsonObject,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> FrozenJsonObject:
        registry = self._registry
        if registry is None:
            raise CapabilityInputError(
                "artifact_snapshot_unavailable",
                "Result snapshot validation is unavailable in this runtime.",
            )
        evidence_call_id = arguments.get("call_id")
        if not isinstance(evidence_call_id, str):
            raise CapabilityInputError(
                "artifact_snapshot_evidence_invalid",
                "A result snapshot requires one exact earlier call ID.",
            )
        try:
            transcript = await self._transcripts.load(run.id)
        except KeyError as error:
            raise CapabilityInputError(
                "artifact_snapshot_evidence_invalid",
                "Result snapshot evidence must belong to the current run.",
            ) from error
        if transcript.run.agent_id != run.agent_id or transcript.run.id != run.id:
            raise CapabilityInputError(
                "artifact_snapshot_evidence_invalid",
                "Result snapshot evidence belongs to another agent or run.",
            )
        current_calls = tuple(
            (message_index, candidate)
            for message_index, message in enumerate(transcript.messages)
            if message.role is MessageRole.ASSISTANT
            for candidate in message.tool_calls
            if candidate.id == call.id
        )
        evidence_calls = tuple(
            (message_index, candidate)
            for message_index, message in enumerate(transcript.messages)
            if message.role is MessageRole.ASSISTANT
            for candidate in message.tool_calls
            if candidate.id == evidence_call_id
        )
        evidence_results = tuple(
            (message_index, block)
            for message_index, message in enumerate(transcript.messages)
            if message.role is MessageRole.TOOL
            for block in message.content
            if isinstance(block, ToolResultBlock) and block.call_id == evidence_call_id
        )
        current_valid = (
            len(current_calls) == 1
            and current_calls[0][1].name == call.name
            and current_calls[0][1].arguments == call.arguments
        )
        ordered_evidence = (
            current_valid
            and len(evidence_calls) == 1
            and len(evidence_results) == 1
            and evidence_calls[0][0] < evidence_results[0][0] < current_calls[0][0]
        )
        if not ordered_evidence or evidence_results[0][1].is_error:
            raise CapabilityInputError(
                "artifact_snapshot_evidence_invalid",
                "Result snapshot evidence must be one earlier successful current-run call.",
                {"call_id": evidence_call_id},
            )
        producer_call = evidence_calls[0][1]
        block = evidence_results[0][1]
        if (
            block.capability_id is None
            or block.executor_id is None
            or block.output_sha256 is None
            or block.sensitivity is None
            or not block.sensitivity_provenance
        ):
            raise CapabilityInputError(
                "artifact_snapshot_evidence_invalid",
                "The selected result lacks validated execution lineage.",
                {"call_id": evidence_call_id},
            )
        try:
            _view, producer_capability = registry.resolve_tool(producer_call.name)
            if (
                producer_capability.id != block.capability_id
                or producer_capability.executor_id != block.executor_id
                or block.output_sha256 != _sha256_json(block.output)
            ):
                raise ValueError("result execution lineage differs")
            registry.validate_arguments(
                producer_capability.id,
                producer_call.arguments,
            )
            if set(block.output) not in (
                {"kind", "data"},
                {"kind", "data", "artifact", "delivery_status"},
            ):
                raise ValueError("result envelope shape differs")
            output_kind = block.output.get("kind")
            result_data = block.output.get("data")
            if not isinstance(output_kind, str) or not isinstance(result_data, Mapping):
                raise ValueError("result data is unavailable")
            registry.validate_output(
                producer_capability.id,
                ToolOutput(
                    kind=output_kind,
                    data=result_data,
                    sensitivity=block.sensitivity,
                    sensitivity_provenance=block.sensitivity_provenance,
                ),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise CapabilityInputError(
                "artifact_snapshot_evidence_invalid",
                "The selected tool result no longer matches its declared contract.",
                {"call_id": evidence_call_id},
            ) from error
        output_provenance = result_data.get("provenance")
        if output_kind == "mcp.read.result" and (
            not isinstance(output_provenance, Mapping)
            or output_provenance.get("output_schema_digest") == "none"
            or not isinstance(result_data.get("structured"), Mapping)
        ):
            raise CapabilityInputError(
                "artifact_snapshot_schema_unavailable",
                "The selected remote result has no admitted structured output schema.",
                {"call_id": evidence_call_id},
            )
        snapshot = serialize_result_snapshot(result_data)
        observed_at = self._clock().astimezone(UTC)
        raw_observed_at = (
            output_provenance.get("observed_at")
            if isinstance(output_provenance, Mapping)
            else result_data.get("observed_at")
        )
        if raw_observed_at is not None:
            if not isinstance(raw_observed_at, str):
                raise CapabilityInputError(
                    "artifact_snapshot_evidence_invalid",
                    "The selected result has an invalid observation time.",
                )
            try:
                observed_at = datetime.fromisoformat(raw_observed_at).astimezone(UTC)
            except ValueError as error:
                raise CapabilityInputError(
                    "artifact_snapshot_evidence_invalid",
                    "The selected result has an invalid observation time.",
                ) from error
        effective_sensitivity = max(
            block.sensitivity,
            request_sensitivity,
            key=lambda item: item.routing_rank,
        )
        prepared: dict[str, object] = dict(arguments)
        prepared.update(
            {
                "_result_data": result_data,
                "_producer_capability_id": producer_capability.id,
                "_producer_executor_id": producer_capability.executor_id,
                "_capability_contract_digest": registry.contract_digest(
                    producer_capability.id
                ),
                "_output_schema_digest": _sha256_json(
                    producer_capability.output_schema
                ),
                "_arguments_sha256": _sha256_json(producer_call.arguments),
                "_result_sha256": snapshot.sha256,
                "_observed_at": observed_at.isoformat(),
                "_result_sensitivity": effective_sensitivity.value,
                "_producer_provenance": {
                    "agent_id": run.agent_id,
                    "run_id": run.id,
                    "result_sensitivity_provenance": block.sensitivity_provenance,
                    "result_output_provenance": (
                        output_provenance
                        if isinstance(output_provenance, Mapping)
                        else {}
                    ),
                },
            }
        )
        return FrozenJsonObject.from_mapping(prepared)

    async def side_effect_plan(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        execution: ToolExecution,
        fingerprint: FrozenJsonObject,
    ) -> SideEffectPlan:
        if capability.id == ARTIFACT_SAVE_LOCAL_CAPABILITY_ID:
            if fingerprint.get("mode") == "replace_bound_file":
                assert self._delivery is not None
                return SideEffectPlan(
                    approval_required=True,
                    approval_arguments=fingerprint,
                    approval_reason=self._delivery.approval_prompt_for_bound_replacement(
                        fingerprint
                    ),
                )
            return SideEffectPlan(
                approval_required=fingerprint.get("requires_approval") is not False,
                approval_arguments=fingerprint,
            )
        if capability.id == ARTIFACT_SET_EXPORT_LOCATION_CAPABILITY_ID:
            assert self._delivery is not None
            return SideEffectPlan(
                approval_arguments=fingerprint,
                approval_reason=self._delivery.approval_prompt_for_default(fingerprint),
            )
        raise ValueError("artifact domain received an unsupported side effect")

    async def finalize_output(
        self,
        run: RunInput,
        call: ToolCall,
        capability: Capability,
        arguments: FrozenJsonObject,
        output: ToolOutput,
        *,
        request_sensitivity: ModelSensitivity,
    ) -> ToolOutput:
        del request_sensitivity
        if output.artifact is not None:
            self._validate_artifact_summary(capability, output)
            output = replace(
                output,
                artifact=await self._bind_provenance(
                    run,
                    capability,
                    arguments,
                    output.artifact,
                ),
            )
        if capability.operational_effect is not OperationalEffect.NONE and not (
            capability.id == ARTIFACT_SAVE_LOCAL_CAPABILITY_ID
            and output.data.get("outcome") == "failed"
        ):
            self._learning.mark_effect_succeeded(run.id)
        if output.sensitivity is not None:
            return output
        source_id = arguments.get("source_id")
        source_ids = (
            (source_id,)
            if isinstance(source_id, str)
            else (() if run.source_id is None else (run.source_id,))
        )
        sensitivity = await self._catalog.admitted_model_sensitivity(
            run.agent_id,
            source_ids,
        )
        if sensitivity is None:
            raise CapabilityInputError(
                "result_classification_unavailable",
                "The current admitted result scope cannot be classified safely.",
                {"capability_id": capability.id},
            )
        readable = await self._catalog.readable_resource_ids(
            run.agent_id,
            source_ids,
        )
        return replace(
            output,
            sensitivity=sensitivity,
            sensitivity_provenance={
                "authority": "artifact_domain_current_scope",
                "capability_id": capability.id,
                "source_ids": source_ids,
                "resource_ids": tuple(sorted(readable)),
            },
        )

    def normalize_error(
        self,
        call: ToolCall,
        error: BaseException,
    ) -> CapabilityFailure | None:
        if isinstance(error, LocalWorkspaceError):
            code = error.code
            if code in {"file_binding_invalid", "file_binding_expired"}:
                code = "artifact_edit_binding_invalid"
            elif code in {"file_too_large", "file_edit_timeout"}:
                code = "artifact_edit_limited"
            return CapabilityFailure(code, error.message, error.details)
        if isinstance(error, ArtifactError):
            return CapabilityFailure(error.code, error.message, error.details)
        return None

    def _validate_artifact_summary(
        self,
        capability: Capability,
        output: ToolOutput,
    ) -> None:
        draft = output.artifact
        assert draft is not None
        if draft.provenance.authorship is ArtifactAuthorship.VALIDATED_TOOL_RESULT:
            result_binding = draft.provenance.result_binding
            valid = (
                capability.id == RESULT_SNAPSHOT_CAPABILITY_ID
                and result_binding is not None
                and output.data.get("filename") == draft.suggested_filename
                and output.data.get("byte_size") == len(draft.content)
                and output.data.get("source_call_id") == result_binding.call_id
                and output.data.get("result_sha256") == result_binding.result_sha256
                and "sha256:" + sha256(draft.content).hexdigest()
                == result_binding.result_sha256
            )
            if not valid:
                raise ToolOutputValidationError(
                    "result snapshot summary differs from its execution provenance"
                )
            return
        if draft.provenance.authorship is not ArtifactAuthorship.EXACT_SOURCE_DATA:
            if capability.id == ARTIFACT_EDIT_TEXT_CAPABILITY_ID:
                local_binding = draft.provenance.local_file_binding
                valid = (
                    local_binding is not None
                    and output.data.get("relative_path") == local_binding.relative_path
                    and output.data.get("filename") == draft.suggested_filename
                    and output.data.get("original_physical_revision")
                    == local_binding.original_physical_revision
                    and output.data.get("observed_content_sha256")
                    == local_binding.observed_content_sha256
                    and output.data.get("source_byte_size")
                    == local_binding.source_byte_size
                    and output.data.get("result_byte_size") == len(draft.content)
                    and output.data.get("change_summary")
                    == FrozenJsonObject.from_mapping(
                        artifact_text_change_summary_to_mapping(
                            local_binding.change_summary
                        )
                    )
                )
                if not valid:
                    raise ToolOutputValidationError(
                        "artifact edit summary differs from its execution provenance"
                    )
            return
        if capability.id in _EXACT_TABULAR_CAPABILITIES:
            valid = (
                output.data.get("format") in {"csv", "xlsx"}
                and output.data.get("filename") == draft.suggested_filename
                and output.data.get("row_count") == draft.provenance.row_count
                and output.data.get("column_count") == len(draft.provenance.columns)
            )
        elif capability.id == ARTIFACT_CONVERT_CAPABILITY_ID:
            valid = (
                output.data.get("source_artifact_id")
                == draft.provenance.derived_from_artifact_id
                and output.data.get("format") == "csv"
                and output.data.get("filename") == draft.suggested_filename
                and output.data.get("row_count") == draft.provenance.row_count
                and output.data.get("column_count") == len(draft.provenance.columns)
            )
        else:
            raise ToolOutputValidationError(
                "exact-source artifact came from a non-export capability"
            )
        if not valid:
            raise ToolOutputValidationError(
                "artifact summary differs from its execution provenance"
            )

    async def _bind_provenance(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
        draft: ArtifactDraft,
    ) -> ArtifactDraft:
        provenance = draft.provenance
        if capability.id == ARTIFACT_EDIT_TEXT_CAPABILITY_ID:
            return self._bind_local_edit(run, arguments, draft)
        if provenance.authorship is ArtifactAuthorship.VALIDATED_TOOL_RESULT:
            if capability.id != RESULT_SNAPSHOT_CAPABILITY_ID:
                raise ToolOutputValidationError(
                    "validated-result artifact came from another capability"
                )
            binding = provenance.result_binding
            result_data = arguments.get("_result_data")
            producer_provenance = arguments.get("_producer_provenance")
            if (
                binding is None
                or not isinstance(result_data, Mapping)
                or not isinstance(producer_provenance, Mapping)
                or draft.content != canonical_json(result_data).encode("utf-8")
                or draft.sensitivity is not binding.result_sensitivity
                or binding.call_id != arguments.get("call_id")
                or binding.capability_id != arguments.get("_producer_capability_id")
                or binding.executor_id != arguments.get("_producer_executor_id")
                or binding.capability_contract_digest
                != arguments.get("_capability_contract_digest")
                or binding.output_schema_digest
                != arguments.get("_output_schema_digest")
                or binding.arguments_sha256 != arguments.get("_arguments_sha256")
                or binding.result_sha256 != arguments.get("_result_sha256")
                or binding.result_sensitivity.value
                != arguments.get("_result_sensitivity")
                or binding.producer_provenance
                != FrozenJsonObject.from_mapping(producer_provenance)
            ):
                raise ToolOutputValidationError(
                    "result snapshot differs from its validated evidence binding"
                )
            return draft
        if provenance.authorship is ArtifactAuthorship.EXACT_SOURCE_DATA:
            if capability.id in _EXACT_TABULAR_CAPABILITIES:
                return await self._bind_exact_export(
                    run,
                    capability,
                    arguments,
                    draft,
                )
            if capability.id == ARTIFACT_CONVERT_CAPABILITY_ID:
                return await self._bind_conversion(run, arguments, draft)
            raise ToolOutputValidationError(
                "exact-source artifact came from a non-export capability"
            )
        if provenance.authorship is not ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS:
            raise ToolOutputValidationError("artifact authorship is not supported")
        if not provenance.evidence_call_ids:
            return replace(
                draft,
                sensitivity=_resolved_sensitivity((draft.sensitivity,)),
                provenance=ArtifactProvenance(
                    authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS
                ),
            )
        try:
            transcript = await self._transcripts.load(run.id)
        except KeyError as error:
            raise CapabilityInputError(
                "invalid_argument_value",
                "Artifact evidence must reference the current run.",
                {"name": "evidence_call_ids"},
            ) from error
        if transcript.run.agent_id != run.agent_id or transcript.run.id != run.id:
            raise CapabilityInputError(
                "invalid_argument_value",
                "Artifact evidence belongs to another run.",
                {"name": "evidence_call_ids"},
            )
        bindings: dict[tuple[str, str], ArtifactResourceBinding] = {}
        sensitivities = [draft.sensitivity]
        schema_cache: dict[str, dict[str, ResourceSchema]] = {}
        for call_id in provenance.evidence_call_ids:
            results = tuple(
                block
                for message in transcript.messages
                if message.role is MessageRole.TOOL
                for block in message.content
                if isinstance(block, ToolResultBlock) and block.call_id == call_id
            )
            call_exists = any(
                candidate.id == call_id
                for message in transcript.messages
                if message.role is MessageRole.ASSISTANT
                for candidate in message.tool_calls
            )
            if len(results) != 1 or not call_exists or results[0].is_error:
                raise CapabilityInputError(
                    "invalid_argument_value",
                    "Artifact evidence must reference one earlier successful data call.",
                    {"name": "evidence_call_ids", "call_id": call_id},
                )
            block = results[0]
            if block.output.get("kind") not in {
                "data.sqlite.query_result",
                "data.postgresql.query_result",
                "data.local_file.read_result",
            }:
                raise CapabilityInputError(
                    "invalid_argument_value",
                    "Artifact evidence must reference a validated data result.",
                    {"name": "evidence_call_ids", "call_id": call_id},
                )
            data = block.output.get("data")
            if not isinstance(data, Mapping):
                raise CapabilityInputError(
                    "invalid_argument_value",
                    "Artifact evidence result data is unavailable.",
                    {"name": "evidence_call_ids", "call_id": call_id},
                )
            if block.sensitivity is not None:
                sensitivities.append(Sensitivity(block.sensitivity.value))
            if block.output.get("kind") == "data.local_file.read_result":
                continue
            source_id = data.get("source_id")
            source_revision = data.get("source_revision")
            if not isinstance(source_id, str) or not isinstance(source_revision, str):
                raise CapabilityInputError(
                    "invalid_argument_value",
                    "Artifact evidence source identity is unavailable.",
                    {"name": "evidence_call_ids", "call_id": call_id},
                )
            raw_revisions = data.get("resource_revisions")
            resources = tuple(
                (resource_id, revision)
                for item in (raw_revisions if isinstance(raw_revisions, tuple) else ())
                if isinstance(item, Mapping)
                and isinstance((resource_id := item.get("resource_id")), str)
                and isinstance((revision := item.get("revision")), str)
            )
            schemas = schema_cache.get(source_id)
            if schemas is None:
                schemas = {
                    item.resource_id: item
                    for item in await self._catalog.resource_schemas(
                        run.agent_id,
                        source_id,
                    )
                }
                schema_cache[source_id] = schemas
            for resource_id, resource_revision in resources:
                schema = schemas.get(resource_id)
                if (
                    schema is None
                    or schema.revision != resource_revision
                    or schema.source_revision != source_revision
                ):
                    raise CapabilityInputError(
                        "invalid_argument_value",
                        "Artifact evidence is no longer current in the catalog.",
                        {"name": "evidence_call_ids", "call_id": call_id},
                    )
                resource_binding = ArtifactResourceBinding(
                    source_id=source_id,
                    source_revision=source_revision,
                    resource_id=resource_id,
                    resource_revision=resource_revision,
                )
                key = (source_id, resource_id)
                if key in bindings and bindings[key] != resource_binding:
                    raise CapabilityInputError(
                        "invalid_argument_value",
                        "Artifact evidence contains conflicting resource revisions.",
                        {"name": "evidence_call_ids", "call_id": call_id},
                    )
                bindings[key] = resource_binding
                try:
                    sensitivities.append(Sensitivity(schema.sensitivity_class))
                except ValueError:
                    sensitivities.append(Sensitivity.RESTRICTED)
        return replace(
            draft,
            sensitivity=_resolved_sensitivity(tuple(sensitivities)),
            provenance=ArtifactProvenance(
                authorship=ArtifactAuthorship.MODEL_AUTHORED_ANALYSIS,
                evidence_call_ids=provenance.evidence_call_ids,
                resource_bindings=tuple(bindings[key] for key in sorted(bindings)),
            ),
        )

    def _bind_local_edit(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
        draft: ArtifactDraft,
    ) -> ArtifactDraft:
        if (
            self._workspace is None
            or run.origin is not RunOrigin.USER
            or run.execution_scope is not None
        ):
            raise ToolOutputValidationError(
                "artifact local edit lacks workspace authority"
            )
        token = arguments.get("binding")
        if not isinstance(token, str):
            raise ToolOutputValidationError(
                "artifact local edit binding is unavailable"
            )
        authenticated = self._workspace.authenticate_file_binding(
            run_id=run.id,
            token=token,
        )
        binding = draft.provenance.local_file_binding
        if (
            binding is None
            or binding.workspace_id != self._workspace.workspace_id
            or binding.workspace_id != authenticated.workspace_id
            or binding.relative_path != authenticated.relative_path
            or binding.original_physical_revision != authenticated.physical_revision
            or binding.source_byte_size != authenticated.size_bytes
            or draft.suggested_filename
            != authenticated.relative_path.rsplit("/", 1)[-1]
            or draft.sensitivity is not Sensitivity(self._workspace.sensitivity.value)
        ):
            raise ToolOutputValidationError(
                "artifact local edit differs from its authenticated source binding"
            )
        return draft

    async def _bind_exact_export(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
        draft: ArtifactDraft,
    ) -> ArtifactDraft:
        source_id = arguments.get("source_id")
        sql = arguments.get("sql")
        parameters = arguments.get("parameters", ())
        if (
            not isinstance(source_id, str)
            or not isinstance(sql, str)
            or not isinstance(parameters, tuple)
        ):
            raise ToolOutputValidationError(
                "exact export execution arguments are unavailable"
            )
        expected_adapter = (
            "postgresql"
            if capability.id == POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID
            else "sqlite"
        )
        if (
            await self._catalog.source_adapter_id(
                run.agent_id,
                source_id,
            )
            != expected_adapter
        ):
            raise _incomplete_export(draft, "catalog_changed")
        resources = await self._catalog.resource_schemas(run.agent_id, source_id)
        try:
            readable = await self._catalog.readable_resource_ids(
                run.agent_id,
                (source_id,),
            )
        except SourcePermissionStateError as error:
            raise _incomplete_export(draft, "permission_state_invalid") from error
        if run.execution_scope is not None:
            readable = readable & frozenset(run.execution_scope.allowed_resource_ids)
        validator = (
            validate_postgresql_read
            if expected_adapter == "postgresql"
            else validate_sqlite_read
        )
        validation = validator(
            sql,
            source_id=source_id,
            resources=resources,
            parameters=parameters,
            allowed_resource_ids=readable,
        )
        if (
            not validation.valid
            or validation.analysis is None
            or validation.source_revision is None
            or not validation.resource_ids
            or len(validation.resource_revisions) != len(validation.resource_ids)
        ):
            raise _incomplete_export(draft, "catalog_changed")
        bindings = tuple(
            ArtifactResourceBinding(
                source_id=source_id,
                source_revision=validation.source_revision,
                resource_id=resource_id,
                resource_revision=revision,
            )
            for resource_id, revision in sorted(validation.resource_revisions)
        )
        parameters_sha256 = (
            "sha256:" + sha256(canonical_json(parameters).encode("utf-8")).hexdigest()
        )
        provenance = draft.provenance
        if (
            provenance.resource_bindings != bindings
            or provenance.sql_fingerprint != validation.analysis.sql_fingerprint
            or provenance.parameters_sha256 != parameters_sha256
        ):
            raise ToolOutputValidationError(
                "exact artifact provenance differs from current runtime facts"
            )
        schemas = {item.resource_id: item for item in resources}
        current: list[Sensitivity] = []
        for resource_id in validation.resource_ids:
            schema = schemas.get(resource_id)
            if schema is None:
                raise ToolOutputValidationError(
                    "exact artifact resource is absent from current catalog facts"
                )
            try:
                current.append(Sensitivity(schema.sensitivity_class))
            except ValueError:
                current.append(Sensitivity.RESTRICTED)
        if draft.sensitivity is not _resolved_sensitivity(tuple(current)):
            raise _incomplete_export(draft, "catalog_changed")
        return replace(
            draft,
            provenance=replace(provenance, resource_bindings=bindings),
        )

    async def _bind_conversion(
        self,
        run: RunInput,
        arguments: Mapping[str, object],
        draft: ArtifactDraft,
    ) -> ArtifactDraft:
        artifact_id = arguments.get("artifact_id")
        if not isinstance(artifact_id, str):
            raise ToolOutputValidationError(
                "artifact conversion input identity is unavailable"
            )
        source = await self._current_ref(run, artifact_id)
        if source is None:
            raise ArtifactError(
                "artifact_missing",
                "The requested artifact is not available in the current conversation.",
                {"artifact_id": artifact_id},
            )
        await self._artifacts.read_ref(source)
        provenance = draft.provenance
        parent = source.provenance
        if (
            provenance.derived_from_artifact_id != source.artifact_id
            or provenance.resource_bindings != parent.resource_bindings
            or provenance.sql_fingerprint != parent.sql_fingerprint
            or provenance.parameters_sha256 != parent.parameters_sha256
            or provenance.columns != parent.columns
            or provenance.row_count != parent.row_count
            or draft.sensitivity is not source.sensitivity
            or draft.media_type != "text/csv"
        ):
            raise ToolOutputValidationError(
                "artifact conversion differs from its verified source snapshot"
            )
        return draft

    async def _current_ref(self, run: RunInput, artifact_id: str) -> ArtifactRef | None:
        return next(
            (
                item
                for item in await self._artifacts.list_refs(
                    conversation_id=run.conversation_id or run.id
                )
                if item.artifact_id == artifact_id
            ),
            None,
        )

    async def _require_readable(
        self,
        run: RunInput,
        source_id: str,
        resource_ids: tuple[str, ...],
    ) -> None:
        try:
            readable = await self._catalog.readable_resource_ids(
                run.agent_id,
                (source_id,),
            )
        except SourcePermissionStateError as error:
            raise CapabilityInputError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from error
        if any(item not in readable for item in resource_ids):
            raise CapabilityInputError(
                "resource_read_not_allowed",
                "The requested resource is not available for reading.",
            )

    async def _validate_sql(
        self,
        run: RunInput,
        capability: Capability,
        arguments: Mapping[str, object],
    ) -> None:
        source_id = arguments.get("source_id")
        sql = arguments.get("sql")
        parameters = arguments.get("parameters", ())
        if (
            not isinstance(source_id, str)
            or not isinstance(sql, str)
            or not sql.strip()
            or not isinstance(parameters, tuple)
        ):
            raise CapabilityInputError(
                "sql_invalid_input",
                "SQL reads require source_id, non-empty sql, and an array of "
                "parameters.",
            )
        expected = (
            "postgresql"
            if capability.id == POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID
            else "sqlite"
        )
        if await self._catalog.source_adapter_id(run.agent_id, source_id) != expected:
            raise CapabilityInputError(
                "sql_source_adapter_mismatch",
                "The selected SQL tool does not match the source adapter.",
                {"expected_adapter": expected, "source_id": source_id},
            )
        schemas = await self._catalog.resource_schemas(run.agent_id, source_id)
        try:
            readable = await self._catalog.readable_resource_ids(
                run.agent_id,
                (source_id,),
            )
        except SourcePermissionStateError as error:
            raise CapabilityInputError(
                "source_permission_state_invalid",
                "Stored source permission state is missing or invalid.",
            ) from error
        if run.execution_scope is not None:
            readable = readable & frozenset(run.execution_scope.allowed_resource_ids)
        validator = (
            validate_postgresql_read
            if expected == "postgresql"
            else validate_sqlite_read
        )
        result = validator(
            sql,
            source_id=source_id,
            resources=schemas,
            parameters=parameters,
            allowed_resource_ids=readable,
        )
        if not result.valid:
            if {item.code for item in result.issues} & {
                "resource_out_of_scope",
                "unknown_resource",
            }:
                raise CapabilityInputError(
                    "resource_read_not_allowed",
                    "One or more requested resources are not available for reading.",
                )
            raise CapabilityInputError(
                "sql_validation_failed",
                "The SQL read is invalid. Correct all reported issues before "
                "retrying.",
                {
                    "issues": tuple(
                        {
                            "code": item.code,
                            "message": item.message,
                            "details": item.details,
                        }
                        for item in result.issues
                    ),
                    "source_id": source_id,
                },
            )


def _incomplete_export(draft: ArtifactDraft, reason: str) -> ArtifactError:
    return ArtifactError(
        "artifact_incomplete_export",
        "Current catalog facts no longer prove the exact artifact.",
        {
            "reason": reason,
            "completed_rows": draft.provenance.row_count or 0,
            "completed_columns": len(draft.provenance.columns),
            "completed_bytes": len(draft.content),
        },
    )


def _resolved_sensitivity(values: tuple[Sensitivity, ...]) -> Sensitivity:
    ordered = {
        Sensitivity.PUBLIC: 0,
        Sensitivity.INTERNAL: 1,
        Sensitivity.CONFIDENTIAL: 2,
        Sensitivity.RESTRICTED: 3,
        Sensitivity.UNKNOWN: 3,
    }
    selected = max(values, key=lambda item: ordered[item])
    return Sensitivity.RESTRICTED if selected is Sensitivity.UNKNOWN else selected


def _sha256_json(value: object) -> str:
    return "sha256:" + sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _tabular_export_capability(
    *,
    capability_id: str,
    executor_id: str,
    adapter_name: str,
) -> Capability:
    return Capability(
        id=capability_id,
        description=(
            f"Run one fresh validated read-only {adapter_name} query and create one "
            "complete exact CSV or XLSX artifact without routing rows through the "
            "model."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "minLength": 1, "maxLength": 256},
                "sql": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_SQL_CHARACTERS,
                },
                "parameters": {
                    "type": "array",
                    "maxItems": MAX_SQL_PARAMETERS,
                },
                "format": {"type": "string", "enum": ["csv", "xlsx"]},
                "filename": {"type": "string", "minLength": 1, "maxLength": 120},
            },
            "required": ["source_id", "sql", "format"],
            "additionalProperties": False,
        },
        output_kind=TABULAR_EXPORT_OUTPUT_KIND,
        output_schema={
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["csv", "xlsx"]},
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
        access_mode=AccessMode.READ,
        automation_eligibility=AutomationEligibility.SCHEDULED_DIRECT,
        artifact_policy=ArtifactPolicy(
            allowed_media_types=frozenset({"text/csv", XLSX_MEDIA_TYPE}),
            allowed_authorships=frozenset({ArtifactAuthorship.EXACT_SOURCE_DATA}),
            allowed_extensions=CSV_ALLOWED_EXTENSIONS + XLSX_ALLOWED_EXTENSIONS,
            artifact_required=True,
            max_artifact_count=1,
            max_bytes_per_artifact=max(MAX_CSV_BYTES, MAX_XLSX_BYTES),
            max_total_bytes_per_call=max(MAX_CSV_BYTES, MAX_XLSX_BYTES),
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
        "mode": "string",
        "outcome": "string",
    }
    optional_types = {
        "workspace_id": "string",
        "relative_path": "string",
        "prior_physical_revision": "string",
        "result_physical_revision": "string",
        "failure_code": "string",
    }
    return {
        "type": "object",
        "properties": {
            name: {"type": kind} for name, kind in {**types, **optional_types}.items()
        },
        "required": list(types),
        "additionalProperties": False,
    }


def _change_summary_schema() -> dict[str, object]:
    integer_names = (
        "operation_count",
        "replacement_count",
        "insertion_count",
        "deletion_count",
        "occurrence_count",
        "bytes_removed",
        "bytes_added",
    )
    return {
        "type": "object",
        "properties": {
            **{name: {"type": "integer", "minimum": 0} for name in integer_names},
            "description": {"type": "string"},
        },
        "required": [*integer_names, "description"],
        "additionalProperties": False,
    }


def _artifact_summary_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "artifact_id": {"type": "string"},
            "filename": {"type": "string"},
            "media_type": {"type": "string"},
            "byte_size": {"type": "integer", "minimum": 0},
            "sha256": {"type": "string"},
            "created_at": {"type": "string"},
            "derived_from_artifact_id": {"type": "string"},
        },
        "required": [
            "artifact_id",
            "filename",
            "media_type",
            "byte_size",
            "sha256",
            "created_at",
        ],
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
    "ARTIFACT_CONVERT_CAPABILITY_ID",
    "ARTIFACT_CONVERT_EXECUTOR_ID",
    "ARTIFACT_CONVERT_OUTPUT_KIND",
    "ARTIFACT_CONVERT_TOOL_NAME",
    "ARTIFACT_DELIVERY_RECEIPT_OUTPUT_KIND",
    "ARTIFACT_DOMAIN_OWNER_ID",
    "ARTIFACT_EXPORT_LOCATION_OUTPUT_KIND",
    "ARTIFACT_LIST_CAPABILITY_ID",
    "ARTIFACT_LIST_EXECUTOR_ID",
    "ARTIFACT_LIST_OUTPUT_KIND",
    "ARTIFACT_LIST_TOOL_NAME",
    "ARTIFACT_READ_CAPABILITY_ID",
    "ARTIFACT_READ_EXECUTOR_ID",
    "ARTIFACT_READ_OUTPUT_KIND",
    "ARTIFACT_READ_TOOL_NAME",
    "ARTIFACT_SAVE_LOCAL_CAPABILITY_ID",
    "ARTIFACT_SAVE_LOCAL_EXECUTOR_ID",
    "ARTIFACT_SAVE_LOCAL_TOOL_NAME",
    "ARTIFACT_SET_EXPORT_LOCATION_CAPABILITY_ID",
    "ARTIFACT_SET_EXPORT_LOCATION_EXECUTOR_ID",
    "ARTIFACT_SET_EXPORT_LOCATION_TOOL_NAME",
    "ArtifactCapabilityDomain",
    "ArtifactDomainCatalog",
    "TABULAR_EXPORT_OUTPUT_KIND",
    "DOCUMENT_CREATE_CAPABILITY_ID",
    "DOCUMENT_CREATE_EXECUTOR_ID",
    "DOCUMENT_CREATE_OUTPUT_KIND",
    "DOCUMENT_CREATE_TOOL_NAME",
    "ExactTabularExportBackend",
    "ExactTabularProgress",
    "ExactTabularExportResult",
    "POSTGRESQL_TABULAR_EXPORT_CAPABILITY_ID",
    "POSTGRESQL_TABULAR_EXPORT_EXECUTOR_ID",
    "POSTGRESQL_TABULAR_EXPORT_TOOL_NAME",
    "PostgreSQLTabularExportExecutor",
    "RESULT_SNAPSHOT_CAPABILITY_ID",
    "RESULT_SNAPSHOT_EXECUTOR_ID",
    "RESULT_SNAPSHOT_OUTPUT_KIND",
    "RESULT_SNAPSHOT_TOOL_NAME",
    "ResultSnapshotExecutor",
    "SQLITE_TABULAR_EXPORT_CAPABILITY_ID",
    "SQLITE_TABULAR_EXPORT_EXECUTOR_ID",
    "SQLITE_TABULAR_EXPORT_TOOL_NAME",
    "SQLiteTabularExportExecutor",
    "artifact_capability_declarations",
    "artifact_declarations",
    "resolved_exact_export_sensitivity",
]
