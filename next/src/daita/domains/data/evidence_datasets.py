"""Authoritative accepted-evidence projection for tabular comparison."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
import re

from ..._json import FrozenJsonObject, canonical_json
from ...adapters.protocols import SourceStore
from ...catalog.models import (
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
)
from ...catalog.protocols import CatalogStore
from ...operations.models import Evidence, Task, TaskStatus
from ...operations.store import OperationStore, VersionedOperation
from ...storage.blobs import BlobMetadata, BlobStore
from .comparison import TabularEvidenceDataset
from .controller import POSTGRESQL_QUERY_EVIDENCE_KIND, SQLITE_QUERY_EVIDENCE_KIND
from .file_capabilities import LOCAL_FILE_READ_EVIDENCE_KIND

_ERROR_CODE = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_SHA256 = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SUPPORTED_KINDS = frozenset(
    {
        LOCAL_FILE_READ_EVIDENCE_KIND,
        POSTGRESQL_QUERY_EVIDENCE_KIND,
        SQLITE_QUERY_EVIDENCE_KIND,
    }
)
_ARTIFACT_MEDIA_TYPE = "application/json"


class DataEvidenceDatasetError(RuntimeError):
    """Normalized fail-closed error at the persisted-evidence trust boundary."""

    def __init__(self, code: str) -> None:
        if not isinstance(code, str) or _ERROR_CODE.fullmatch(code) is None:
            raise ValueError("dataset error code is invalid")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True, slots=True)
class _ParsedPayload:
    evidence_kind: str
    source_id: str
    source_revision: str
    resource_revisions: tuple[tuple[str, str], ...]
    columns: tuple[str, ...]
    rows: tuple[FrozenJsonObject, ...]
    total_rows: int
    complete: bool
    truncation_reasons: tuple[str, ...]
    row_limit: int
    byte_limit: int


class PersistedAcceptedEvidenceDatasetReader:
    """Load comparison inputs only from current accepted operation evidence."""

    def __init__(
        self,
        operation_store: OperationStore,
        catalog_store: CatalogStore,
        source_store: SourceStore,
        blob_store: BlobStore,
        *,
        max_rows: int = 10_000,
        max_columns: int = 512,
        max_blob_bytes: int = 16 * 1_024 * 1_024,
    ) -> None:
        for owner, methods, name in (
            (operation_store, ("load",), "operation_store"),
            (
                catalog_store,
                ("load_resource", "load_revision", "load_sync"),
                "catalog_store",
            ),
            (source_store, ("load_source",), "source_store"),
            (blob_store, ("metadata", "open"), "blob_store"),
        ):
            if any(not callable(getattr(owner, method, None)) for method in methods):
                raise TypeError(f"{name} does not provide its required methods")
        for value, name, minimum in (
            (max_rows, "max_rows", 1),
            (max_columns, "max_columns", 1),
            (max_blob_bytes, "max_blob_bytes", 2),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if max_rows > 10_000:
            raise ValueError("max_rows cannot exceed the comparison dataset bound")
        if max_columns > 512:
            raise ValueError("max_columns cannot exceed the comparison column bound")
        self._operation_store = operation_store
        self._catalog_store = catalog_store
        self._source_store = source_store
        self._blob_store = blob_store
        self._max_rows = max_rows
        self._max_columns = max_columns
        self._max_blob_bytes = max_blob_bytes

    async def load_dataset(
        self,
        *,
        operation_id: str,
        evidence_id: str,
    ) -> TabularEvidenceDataset:
        _requested_identity(operation_id, "operation_id")
        _requested_identity(evidence_id, "evidence_id")
        versioned = await self._load_operation(operation_id)
        snapshot = versioned.snapshot
        if snapshot.operation.id != operation_id:
            raise DataEvidenceDatasetError("operation_scope_mismatch")
        evidence = next(
            (item for item in snapshot.evidence if item.id == evidence_id),
            None,
        )
        if evidence is None:
            raise DataEvidenceDatasetError("evidence_not_found")
        if evidence.operation_id != operation_id:
            raise DataEvidenceDatasetError("evidence_scope_mismatch")
        if not evidence.accepted:
            raise DataEvidenceDatasetError("evidence_not_accepted")
        if evidence.kind not in _SUPPORTED_KINDS:
            raise DataEvidenceDatasetError("evidence_kind_unsupported")
        if evidence.schema_version != 1:
            raise DataEvidenceDatasetError("evidence_schema_unsupported")
        task = next(
            (item for item in snapshot.tasks if item.id == evidence.task_id), None
        )
        self._validate_current_task(task, evidence)
        parsed = self._parse_payload(evidence)
        await self._validate_current_provenance(
            agent_id=snapshot.operation.agent_id,
            parsed=parsed,
        )

        rows = parsed.rows
        sensitivity_class = "internal"
        retention_class = "operation"
        if evidence.blob_id is None:
            expected_hash = (
                "sha256:"
                + sha256(canonical_json(evidence.payload).encode("utf-8")).hexdigest()
            )
            if evidence.content_hash != expected_hash:
                raise DataEvidenceDatasetError("evidence_integrity_failed")
        else:
            blob_rows, metadata = await self._load_blob_rows(
                evidence,
                columns=parsed.columns,
            )
            if len(blob_rows) < len(rows) or blob_rows[: len(rows)] != rows:
                raise DataEvidenceDatasetError("blob_payload_mismatch")
            if (
                len(blob_rows) > parsed.row_limit
                or len(blob_rows) > parsed.total_rows
                or (parsed.complete and blob_rows != rows)
            ):
                raise DataEvidenceDatasetError("dataset_bounds_exceeded")
            rows = blob_rows
            sensitivity_class = metadata.sensitivity_class
            retention_class = metadata.retention_class

        try:
            return TabularEvidenceDataset(
                operation_id=operation_id,
                evidence_id=evidence.id,
                evidence_kind=evidence.kind,
                source_id=parsed.source_id,
                source_revision=parsed.source_revision,
                resource_revisions=parsed.resource_revisions,
                columns=parsed.columns,
                rows=rows,
                complete=parsed.complete,
                truncation_reasons=parsed.truncation_reasons,
                row_limit=parsed.row_limit,
                byte_limit=parsed.byte_limit,
                sensitivity_class=sensitivity_class,
                retention_class=retention_class,
            )
        except (TypeError, ValueError) as error:
            raise DataEvidenceDatasetError("dataset_invalid") from error

    async def _load_operation(self, operation_id: str) -> VersionedOperation:
        try:
            versioned = await self._operation_store.load(operation_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise DataEvidenceDatasetError("operation_unavailable") from error
        if not isinstance(versioned, VersionedOperation):
            raise DataEvidenceDatasetError("operation_invalid")
        return versioned

    @staticmethod
    def _validate_current_task(task: Task | None, evidence: Evidence) -> None:
        if task is None:
            raise DataEvidenceDatasetError("evidence_task_missing")
        if (
            task.operation_id != evidence.operation_id
            or task.turn_id != evidence.turn_id
            or task.capability_id != evidence.capability_id
            or task.executor_id != evidence.executor_id
        ):
            raise DataEvidenceDatasetError("evidence_task_mismatch")
        if (
            task.status is not TaskStatus.SUCCEEDED
            or task.attempt != evidence.attempt
            or evidence.id not in task.evidence_ids
        ):
            raise DataEvidenceDatasetError("evidence_attempt_not_current")

    def _parse_payload(self, evidence: Evidence) -> _ParsedPayload:
        try:
            payload = evidence.payload
            source_id = _text_field(payload, "source_id", maximum=512)
            source_revision = _text_field(
                payload,
                "source_revision",
                maximum=1_024,
            )
            resource_revisions = _parse_resource_revisions(evidence.kind, payload)
            columns = _text_sequence(
                payload.get("columns"),
                "columns",
                maximum_items=self._max_columns,
                maximum_length=256,
            )
            rows = _row_sequence(
                payload.get("rows"),
                columns=columns,
                maximum_items=self._max_rows,
            )
            returned_rows = _bounded_integer(
                payload.get("returned_rows"),
                "returned_rows",
                minimum=0,
                maximum=self._max_rows,
            )
            if returned_rows != len(rows):
                raise ValueError("returned_rows does not match rows")
            total_rows = _bounded_integer(
                payload.get("total_rows"),
                "total_rows",
                minimum=0,
                maximum=2**63 - 1,
            )
            if total_rows < returned_rows:
                raise ValueError("total_rows is below returned_rows")
            row_limit = _bounded_integer(
                payload.get("row_limit"),
                "row_limit",
                minimum=1,
                maximum=self._max_rows,
            )
            if len(rows) > row_limit:
                raise ValueError("rows exceed row_limit")
            byte_limit = _bounded_integer(
                payload.get("byte_limit"),
                "byte_limit",
                minimum=2,
                maximum=self._max_blob_bytes,
            )
            utf8_bytes = _bounded_integer(
                payload.get("utf8_bytes"),
                "utf8_bytes",
                minimum=2,
                maximum=byte_limit,
            )
            if len(canonical_json(rows).encode("utf-8")) != utf8_bytes:
                raise ValueError("utf8_bytes does not match rows")
            truncated = _boolean_field(payload, "truncated")
            reasons = _text_sequence(
                payload.get("truncation_reasons"),
                "truncation_reasons",
                maximum_items=16,
                maximum_length=128,
                allow_empty=True,
            )
            if truncated != bool(reasons):
                raise ValueError("truncated does not match truncation_reasons")
            if not truncated and total_rows != returned_rows:
                raise ValueError("complete projection row counts disagree")
            if payload.get("trust_classification") != "untrusted_external_data":
                raise ValueError("trust classification is invalid")
            input_complete = True
            if evidence.kind == LOCAL_FILE_READ_EVIDENCE_KIND:
                input_complete = _boolean_field(payload, "complete")
                if not input_complete and not truncated:
                    raise ValueError("incomplete file read must be truncated")
            complete = input_complete and not truncated
            if not complete and not reasons:
                raise ValueError("incomplete dataset requires truncation reasons")
            return _ParsedPayload(
                evidence_kind=evidence.kind,
                source_id=source_id,
                source_revision=source_revision,
                resource_revisions=resource_revisions,
                columns=columns,
                rows=rows,
                total_rows=total_rows,
                complete=complete,
                truncation_reasons=reasons,
                row_limit=row_limit,
                byte_limit=byte_limit,
            )
        except DataEvidenceDatasetError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise DataEvidenceDatasetError("evidence_payload_invalid") from error

    async def _validate_current_provenance(
        self,
        *,
        agent_id: str,
        parsed: _ParsedPayload,
    ) -> None:
        try:
            source = await self._source_store.load_source(agent_id, parsed.source_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise DataEvidenceDatasetError("source_unavailable") from error
        if (
            source is None
            or source.agent_id != agent_id
            or source.id != parsed.source_id
            or not source.active
        ):
            raise DataEvidenceDatasetError("source_inactive")
        expected_adapter_id = {
            LOCAL_FILE_READ_EVIDENCE_KIND: "local-directory",
            POSTGRESQL_QUERY_EVIDENCE_KIND: "postgresql",
            SQLITE_QUERY_EVIDENCE_KIND: "sqlite",
        }[parsed.evidence_kind]
        if source.adapter_id != expected_adapter_id:
            raise DataEvidenceDatasetError("source_adapter_mismatch")

        for resource_id, expected_revision in parsed.resource_revisions:
            resource = await self._load_catalog_resource(agent_id, resource_id)
            if resource.source_id != parsed.source_id:
                raise DataEvidenceDatasetError("catalog_resource_scope_mismatch")
            if resource.current_revision != expected_revision:
                raise DataEvidenceDatasetError("catalog_resource_stale")
            revision = await self._load_catalog_revision(
                agent_id,
                resource_id,
                expected_revision,
            )
            if (
                revision.resource_id != resource_id
                or revision.revision != expected_revision
                or revision.sync_id != resource.current_sync_id
            ):
                raise DataEvidenceDatasetError("catalog_revision_mismatch")
            if revision.source_revision != parsed.source_revision:
                raise DataEvidenceDatasetError("catalog_source_stale")
            sync = await self._load_catalog_sync(
                agent_id,
                resource,
            )
            if (
                sync.status is not CatalogSyncStatus.SUCCEEDED
                or sync.source_id != parsed.source_id
                or sync.adapter_id != expected_adapter_id
            ):
                raise DataEvidenceDatasetError("catalog_source_stale")
            if (
                parsed.evidence_kind
                in {SQLITE_QUERY_EVIDENCE_KIND, POSTGRESQL_QUERY_EVIDENCE_KIND}
                and sync.source_revision != parsed.source_revision
            ):
                raise DataEvidenceDatasetError("catalog_source_stale")

    async def _load_catalog_resource(
        self,
        agent_id: str,
        resource_id: str,
    ) -> CatalogResource:
        try:
            resource = await self._catalog_store.load_resource(agent_id, resource_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise DataEvidenceDatasetError("catalog_unavailable") from error
        if not isinstance(resource, CatalogResource):
            raise DataEvidenceDatasetError("catalog_resource_missing")
        if resource.agent_id != agent_id or resource.id != resource_id:
            raise DataEvidenceDatasetError("catalog_resource_scope_mismatch")
        return resource

    async def _load_catalog_revision(
        self,
        agent_id: str,
        resource_id: str,
        revision: str,
    ) -> CatalogResourceRevision:
        try:
            record = await self._catalog_store.load_revision(
                agent_id,
                resource_id,
                revision,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise DataEvidenceDatasetError("catalog_unavailable") from error
        if not isinstance(record, CatalogResourceRevision):
            raise DataEvidenceDatasetError("catalog_revision_missing")
        return record

    async def _load_catalog_sync(
        self,
        agent_id: str,
        resource: CatalogResource,
    ) -> CatalogSync:
        try:
            sync = await self._catalog_store.load_sync(
                agent_id,
                resource.current_sync_id,
            )
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise DataEvidenceDatasetError("catalog_unavailable") from error
        if (
            not isinstance(sync, CatalogSync)
            or sync.agent_id != agent_id
            or sync.id != resource.current_sync_id
        ):
            raise DataEvidenceDatasetError("catalog_sync_missing")
        return sync

    async def _load_blob_rows(
        self,
        evidence: Evidence,
        *,
        columns: tuple[str, ...],
    ) -> tuple[tuple[FrozenJsonObject, ...], BlobMetadata]:
        assert evidence.blob_id is not None
        try:
            metadata = await self._blob_store.metadata(evidence.blob_id)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise DataEvidenceDatasetError("blob_unavailable") from error
        if not isinstance(metadata, BlobMetadata):
            raise DataEvidenceDatasetError("blob_unavailable")
        if (
            metadata.blob_id != evidence.blob_id
            or metadata.operation_id != evidence.operation_id
            or metadata.task_id != evidence.task_id
            or metadata.evidence_id != evidence.id
            or metadata.digest != evidence.content_hash
            or metadata.media_type != _ARTIFACT_MEDIA_TYPE
            or metadata.tombstoned_at is not None
            or metadata.deleted_at is not None
        ):
            raise DataEvidenceDatasetError("blob_metadata_invalid")
        if metadata.size_bytes > self._max_blob_bytes:
            raise DataEvidenceDatasetError("dataset_bounds_exceeded")
        content = await self._read_blob_content(metadata)
        actual_digest = "sha256:" + sha256(content).hexdigest()
        if len(content) != metadata.size_bytes or actual_digest != metadata.digest:
            raise DataEvidenceDatasetError("blob_integrity_failed")
        try:
            decoded = content.decode("utf-8")
            value = json.loads(
                decoded,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
            if canonical_json(value).encode("utf-8") != content:
                raise ValueError("artifact is not canonical JSON")
            rows = _row_sequence(
                value,
                columns=columns,
                maximum_items=self._max_rows,
            )
        except (
            TypeError,
            UnicodeDecodeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            raise DataEvidenceDatasetError("blob_payload_invalid") from error
        return rows, metadata

    async def _read_blob_content(self, metadata: BlobMetadata) -> bytes:
        try:
            reader = await self._blob_store.open(metadata.blob_id)
            async with reader:
                if reader.metadata != metadata:
                    raise DataEvidenceDatasetError("blob_metadata_invalid")
                chunks: list[bytes] = []
                total = 0
                while total <= metadata.size_bytes:
                    chunk = await reader.read(
                        min(65_536, metadata.size_bytes + 1 - total)
                    )
                    if not isinstance(chunk, bytes):
                        raise DataEvidenceDatasetError("blob_integrity_failed")
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
                    if total > metadata.size_bytes:
                        raise DataEvidenceDatasetError("blob_integrity_failed")
                return b"".join(chunks)
        except asyncio.CancelledError:
            raise
        except DataEvidenceDatasetError:
            raise
        except Exception as error:
            raise DataEvidenceDatasetError("blob_unavailable") from error


def _requested_identity(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise DataEvidenceDatasetError("request_invalid")
    if len(value) > 512:
        raise DataEvidenceDatasetError("request_invalid")


def _text_field(
    payload: Mapping[str, object],
    field_name: str,
    *,
    maximum: int,
) -> str:
    value = payload.get(field_name)
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > maximum
    ):
        raise ValueError(f"{field_name} is invalid")
    return value


def _boolean_field(payload: Mapping[str, object], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _bounded_integer(
    value: object,
    field_name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not minimum <= value <= maximum
    ):
        raise ValueError(f"{field_name} is outside its bound")
    return value


def _text_sequence(
    value: object,
    field_name: str,
    *,
    maximum_items: int,
    maximum_length: int,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence")
    result = tuple(value)
    if not allow_empty and not result:
        raise ValueError(f"{field_name} cannot be empty")
    if len(result) > maximum_items:
        raise ValueError(f"{field_name} exceeds its item bound")
    if any(
        not isinstance(item, str)
        or not item.strip()
        or item != item.strip()
        or len(item) > maximum_length
        for item in result
    ):
        raise ValueError(f"{field_name} contains invalid text")
    if len(result) != len(set(result)):
        raise ValueError(f"{field_name} contains duplicates")
    return result


def _row_sequence(
    value: object,
    *,
    columns: tuple[str, ...] | None,
    maximum_items: int,
) -> tuple[FrozenJsonObject, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError("rows must be a sequence")
    raw_rows = tuple(value)
    if len(raw_rows) > maximum_items:
        raise ValueError("rows exceed their item bound")
    known_columns = None if columns is None else set(columns)
    rows: list[FrozenJsonObject] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            raise TypeError("rows must contain objects")
        frozen = FrozenJsonObject.from_mapping(row)
        if known_columns is not None and not set(frozen).issubset(known_columns):
            raise ValueError("row contains a column outside the declared schema")
        rows.append(frozen)
    return tuple(rows)


def _parse_resource_revisions(
    evidence_kind: str,
    payload: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    if evidence_kind == LOCAL_FILE_READ_EVIDENCE_KIND:
        resource_id = _text_field(payload, "resource_id", maximum=512)
        revision = _text_field(payload, "resource_revision", maximum=71)
        if _SHA256.fullmatch(revision) is None:
            raise ValueError("resource_revision must use sha256")
        return ((resource_id, revision),)
    raw = payload.get("resource_revisions")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise TypeError("resource_revisions must be a sequence")
    values: list[tuple[str, str]] = []
    for item in raw:
        if not isinstance(item, Mapping) or set(item) != {"resource_id", "revision"}:
            raise ValueError("resource revision entry is invalid")
        resource_id = _text_field(item, "resource_id", maximum=512)
        revision = _text_field(item, "revision", maximum=71)
        if _SHA256.fullmatch(revision) is None:
            raise ValueError("resource revision must use sha256")
        values.append((resource_id, revision))
    if not values or len(values) > 1_000:
        raise ValueError("resource_revisions count is invalid")
    result = tuple(sorted(values))
    if len({resource_id for resource_id, _ in result}) != len(result):
        raise ValueError("resource_revisions repeat a resource")
    resource_ids = _text_sequence(
        payload.get("resource_ids"),
        "resource_ids",
        maximum_items=1_000,
        maximum_length=512,
    )
    if set(resource_ids) != {resource_id for resource_id, _ in result}:
        raise ValueError("resource_ids do not match resource_revisions")
    return result


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON object key")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


__all__ = [
    "DataEvidenceDatasetError",
    "PersistedAcceptedEvidenceDatasetReader",
]
