from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from hashlib import sha256
from types import SimpleNamespace
from typing import cast

import pytest

from daita._json import FrozenJsonObject, canonical_json
from daita.adapters.models import SourceRegistration
from daita.adapters.protocols import SourceStore
from daita.catalog import (
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    ResourceKind,
    Sensitivity,
    catalog_resource_id,
)
from daita.catalog.protocols import CatalogStore
from daita.domains.data.evidence_datasets import (
    DataEvidenceDatasetError,
    PersistedAcceptedEvidenceDatasetReader,
)
from daita.events.models import RuntimeEvent
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
    ToolDefinition,
)
from daita.loop.models import LoopBudgets, LoopPhase, LoopState, Turn
from daita.operations.checkpoints import (
    ModelCall,
    ModelCallStatus,
    OperationSnapshot,
)
from daita.operations.models import (
    AgentTrigger,
    Evidence,
    Operation,
    OperationStatus,
    Task,
    TaskStatus,
    TriggerKind,
)
from daita.operations.store import OperationStore, VersionedOperation
from daita.storage.blobs import BlobMetadata, BlobStore

NOW = datetime(2026, 7, 18, 13, 0, tzinfo=timezone.utc)
RESOURCE_REVISION = "sha256:" + "a" * 64
OTHER_REVISION = "sha256:" + "b" * 64


class OperationStoreStub:
    def __init__(self, operation: VersionedOperation) -> None:
        self.operation = operation

    async def load(self, _operation_id: str) -> VersionedOperation:
        return self.operation


class CatalogStoreStub:
    def __init__(
        self,
        resource: CatalogResource,
        revision: CatalogResourceRevision,
        sync: CatalogSync,
    ) -> None:
        self.resource = resource
        self.revision = revision
        self.sync = sync

    async def load_resource(
        self,
        _agent_id: str,
        _resource_id: str,
    ) -> CatalogResource:
        return self.resource

    async def load_revision(
        self,
        _agent_id: str,
        _resource_id: str,
        _revision: str,
    ) -> CatalogResourceRevision:
        return self.revision

    async def load_sync(self, _agent_id: str, _sync_id: str) -> CatalogSync:
        return self.sync


class SourceStoreStub:
    def __init__(self, source: SourceRegistration) -> None:
        self.source = source

    async def load_source(
        self,
        _agent_id: str,
        _source_id: str,
    ) -> SourceRegistration:
        return self.source


class BlobReaderStub:
    def __init__(self, metadata: BlobMetadata, content: bytes) -> None:
        self._metadata = metadata
        self._content = content
        self._offset = 0

    @property
    def metadata(self) -> BlobMetadata:
        return self._metadata

    async def read(self, size: int) -> bytes:
        chunk = self._content[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    async def close(self) -> None:
        return None

    async def __aenter__(self) -> BlobReaderStub:
        return self

    async def __aexit__(self, *_args: object) -> None:
        await self.close()


class BlobStoreStub:
    def __init__(
        self,
        metadata: BlobMetadata | None = None,
        content: bytes = b"",
    ) -> None:
        self._metadata = metadata
        self._content = content

    async def metadata(self, _blob_id: str) -> BlobMetadata | None:
        return self._metadata

    async def open(self, _blob_id: str) -> BlobReaderStub:
        assert self._metadata is not None
        return BlobReaderStub(self._metadata, self._content)


@dataclass(slots=True)
class ReaderCase:
    reader: PersistedAcceptedEvidenceDatasetReader
    operation: VersionedOperation
    catalog: CatalogStoreStub
    source: SourceStoreStub
    blob: BlobStoreStub
    payload: dict[str, object]
    evidence_id: str = "evidence-1"


def _frozen_rows(rows: tuple[object, ...]) -> tuple[FrozenJsonObject, ...]:
    assert all(isinstance(row, FrozenJsonObject) for row in rows)
    return cast(tuple[FrozenJsonObject, ...], rows)


def _projection_payload(
    rows: list[dict[str, object]],
    *,
    total_rows: int | None = None,
    truncated: bool = False,
    reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "rows": rows,
        "total_rows": len(rows) if total_rows is None else total_rows,
        "returned_rows": len(rows),
        "utf8_bytes": len(canonical_json(rows).encode("utf-8")),
        "row_limit": 100,
        "byte_limit": 65_536,
        "truncated": truncated,
        "truncation_reasons": reasons or [],
        "trust_classification": "untrusted_external_data",
    }


def _snapshot(
    *,
    payload: dict[str, object],
    evidence_kind: str,
    content_hash: str,
    blob_id: str | None,
    accepted: bool = True,
) -> VersionedOperation:
    trigger = AgentTrigger(
        id="trigger-1",
        agent_id="agent-1",
        kind=TriggerKind.USER,
        source_id="user-1",
        payload={"message": "Compare customer exports."},
        created_at=NOW,
    )
    operation = Operation(
        id="operation-1",
        agent_id=trigger.agent_id,
        trigger_id=trigger.id,
        status=OperationStatus.RUNNING,
        created_at=NOW,
        updated_at=NOW,
    )
    capability_id = (
        "data.file.read"
        if evidence_kind == "data.file.read_result"
        else "data.sqlite.query"
    )
    executor_id = capability_id + ".executor"
    request = ModelRequest(
        operation_id=operation.id,
        turn_id="turn-1",
        messages=(
            CanonicalMessage(
                agent_id=operation.agent_id,
                operation_id=operation.id,
                turn_id="turn-1",
                role=MessageRole.USER,
                content=(TextBlock("Read customers."),),
            ),
        ),
        tools=(
            ToolDefinition(
                name="read_customers",
                description="Read bounded customer rows.",
                input_schema={"type": "object"},
            ),
        ),
    )
    response = ModelResponse(
        finish_reason=FinishReason.TOOL_CALLS,
        tool_calls=(ToolCall(id="call-1", name="read_customers", arguments={}),),
    )
    model_call = ModelCall(
        id="model-call-1",
        operation_id=operation.id,
        turn_id="turn-1",
        provider_id="mock:scripted",
        request=request,
        response=response,
        status=ModelCallStatus.COMPLETED,
        created_at=NOW,
        updated_at=NOW,
    )
    evidence = Evidence(
        id="evidence-1",
        operation_id=operation.id,
        task_id="task-1",
        turn_id="turn-1",
        capability_id=capability_id,
        executor_id=executor_id,
        kind=evidence_kind,
        schema_version=1,
        attempt=1,
        accepted=accepted,
        payload=payload,
        content_hash=content_hash,
        created_at=NOW,
        blob_id=blob_id,
    )
    task = Task(
        id="task-1",
        operation_id=operation.id,
        turn_id="turn-1",
        call_id="call-1",
        capability_id=capability_id,
        executor_id=executor_id,
        status=TaskStatus.SUCCEEDED if accepted else TaskStatus.FAILED,
        attempt=1,
        arguments={},
        created_at=NOW,
        updated_at=NOW,
        evidence_ids=(evidence.id,) if accepted else (),
        error_code=None if accepted else "evidence_rejected",
    )
    events = tuple(
        RuntimeEvent(
            id=f"event-{index}",
            type=event_type,
            agent_id=operation.agent_id,
            operation_id=operation.id,
            payload={},
            created_at=NOW,
        )
        for index, event_type in enumerate(
            ("trigger.received", "operation.created"),
            start=1,
        )
    )
    snapshot = OperationSnapshot(
        trigger=trigger,
        operation=operation,
        loop_state=LoopState(
            phase=LoopPhase.OBSERVING,
            turn_count=1,
            action_count=1,
        ),
        budgets=LoopBudgets(),
        turns=(
            Turn(
                id="turn-1",
                operation_id=operation.id,
                number=1,
                model_request_id=model_call.id,
                model_response_id=model_call.id,
                created_at=NOW,
            ),
        ),
        model_calls=(model_call,),
        readiness=(),
        tasks=(task,),
        evidence=(evidence,),
        observations=(),
        events=events,
    )
    return VersionedOperation(snapshot=snapshot, revision=1)


def _case(
    *,
    file_evidence: bool = False,
    rows: list[dict[str, object]] | None = None,
    total_rows: int | None = None,
    truncated: bool = False,
    reasons: list[str] | None = None,
    accepted: bool = True,
    blob_content: bytes | None = None,
    opened_blob_content: bytes | None = None,
) -> ReaderCase:
    inline_rows = rows or [{"id": 1, "name": "Ada"}]
    adapter_id = "local-directory" if file_evidence else "sqlite"
    native_identity = "/allowed" if file_evidence else "/data/customers.sqlite3"
    source = SourceRegistration.build(
        agent_id="agent-1",
        adapter_id=adapter_id,
        native_identity=native_identity,
        display_name="Customers",
        configuration={},
        attached_at=NOW,
    )
    source_revision = "sha256:" + "c" * 64 if file_evidence else "schema_version:1"
    resource_kind = ResourceKind.FILE if file_evidence else ResourceKind.TABLE
    resource_native_identity = (
        "exports/customers.csv" if file_evidence else "main.customers"
    )
    resource_id = catalog_resource_id(
        source.id,
        resource_kind,
        resource_native_identity,
    )
    payload = {
        **_projection_payload(
            inline_rows,
            total_rows=total_rows,
            truncated=truncated,
            reasons=reasons,
        ),
        "columns": ["id", "name"],
        "source_id": source.id,
        "source_revision": source_revision,
    }
    evidence_kind = (
        "data.file.read_result" if file_evidence else "data.sqlite.query_result"
    )
    if file_evidence:
        payload.update(
            {
                "complete": True,
                "encoding": "utf-8",
                "format": "csv",
                "resource_id": resource_id,
                "resource_revision": RESOURCE_REVISION,
            }
        )
    else:
        payload.update(
            {
                "resource_ids": [resource_id],
                "resource_revisions": [
                    {
                        "resource_id": resource_id,
                        "revision": RESOURCE_REVISION,
                    }
                ],
            }
        )
    revision = CatalogResourceRevision.build(
        resource_id=resource_id,
        sync_id="sync-1",
        observed_at=NOW,
        source_revision=source_revision,
    )
    resource = CatalogResource.build(
        agent_id="agent-1",
        source_id=source.id,
        native_identity=resource_native_identity,
        external_uri=f"test://{resource_native_identity}",
        kind=resource_kind,
        name="customers",
        sensitivity=Sensitivity.INTERNAL,
        revision=revision,
        first_observed_at=NOW,
        last_observed_at=NOW,
    )
    # The resource revision is content-addressed by its catalog components. The
    # fixture uses that real revision in evidence rather than the display canary.
    payload_revision = resource.current_revision
    if file_evidence:
        payload["resource_revision"] = payload_revision
    else:
        payload["resource_revisions"] = [
            {"resource_id": resource_id, "revision": payload_revision}
        ]
    sync_source_revision = "manifest:" + "d" * 64 if file_evidence else source_revision
    sync = CatalogSync(
        id="sync-1",
        agent_id="agent-1",
        source_id=source.id,
        adapter_id=adapter_id,
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=NOW,
        completed_at=NOW,
        source_revision=sync_source_revision,
        resource_count=1,
    )

    blob_id = None
    metadata = None
    if blob_content is None:
        content_hash = (
            "sha256:" + sha256(canonical_json(payload).encode("utf-8")).hexdigest()
        )
    else:
        blob_id = "blob-1"
        content_hash = "sha256:" + sha256(blob_content).hexdigest()
        metadata = BlobMetadata(
            blob_id=blob_id,
            digest=content_hash,
            size_bytes=len(blob_content),
            media_type="application/json",
            created_at=NOW,
            sensitivity_class="confidential",
            retention_class="operation",
            operation_id="operation-1",
            task_id="task-1",
            evidence_id="evidence-1",
        )
    operation = _snapshot(
        payload=payload,
        evidence_kind=evidence_kind,
        content_hash=content_hash,
        blob_id=blob_id,
        accepted=accepted,
    )
    catalog = CatalogStoreStub(resource, revision, sync)
    source_store = SourceStoreStub(source)
    opened_content = (
        blob_content if opened_blob_content is None else opened_blob_content
    )
    blob_store = BlobStoreStub(
        metadata, b"" if opened_content is None else opened_content
    )
    reader = PersistedAcceptedEvidenceDatasetReader(
        cast(OperationStore, OperationStoreStub(operation)),
        cast(CatalogStore, catalog),
        cast(SourceStore, source_store),
        cast(BlobStore, blob_store),
    )
    return ReaderCase(
        reader=reader,
        operation=operation,
        catalog=catalog,
        source=source_store,
        blob=blob_store,
        payload=payload,
    )


async def test_loads_current_inline_sqlite_evidence() -> None:
    case = _case()

    dataset = await case.reader.load_dataset(
        operation_id="operation-1",
        evidence_id=case.evidence_id,
    )

    assert dataset.evidence_kind == "data.sqlite.query_result"
    assert dataset.columns == ("id", "name")
    assert tuple(row.to_dict() for row in _frozen_rows(dataset.rows)) == (
        {"id": 1, "name": "Ada"},
    )
    assert dataset.complete is True
    assert dataset.resource_revisions == (
        (case.catalog.resource.id, case.catalog.resource.current_revision),
    )


async def test_blob_file_rows_require_prefix_and_remain_explicitly_partial() -> None:
    artifact_rows = [
        {"id": 1, "name": "Ada"},
        {"id": 2, "name": "Grace"},
    ]
    blob_content = canonical_json(artifact_rows).encode("utf-8")
    case = _case(
        file_evidence=True,
        rows=artifact_rows[:1],
        total_rows=2,
        truncated=True,
        reasons=["byte_limit"],
        blob_content=blob_content,
    )

    dataset = await case.reader.load_dataset(
        operation_id="operation-1",
        evidence_id=case.evidence_id,
    )

    assert tuple(row.to_dict() for row in _frozen_rows(dataset.rows)) == tuple(
        artifact_rows
    )
    assert dataset.complete is False
    assert dataset.truncation_reasons == ("byte_limit",)
    assert dataset.sensitivity_class == "confidential"
    # File evidence is fresh against its resource revision even though the
    # successful source sync carries the distinct directory-manifest revision.
    assert case.catalog.sync.source_revision != dataset.source_revision


@pytest.mark.parametrize(
    ("mutation", "expected_code"),
    [
        ("rejected", "evidence_not_accepted"),
        ("detached", "source_inactive"),
        ("adapter", "source_adapter_mismatch"),
        ("resource_revision", "catalog_resource_stale"),
        ("source_revision", "catalog_source_stale"),
        ("payload_count", "evidence_payload_invalid"),
        ("payload_hash", "evidence_integrity_failed"),
    ],
)
async def test_invalid_inline_evidence_fails_closed(
    mutation: str,
    expected_code: str,
) -> None:
    case = _case(accepted=mutation != "rejected")
    if mutation == "detached":
        case.source.source = case.source.source.detach(NOW)
    elif mutation == "adapter":
        current = case.source.source
        case.source.source = SimpleNamespace(  # type: ignore[assignment]
            id=current.id,
            agent_id=current.agent_id,
            adapter_id="local-directory",
            active=True,
        )
    elif mutation == "resource_revision":
        case.catalog.resource = replace(
            case.catalog.resource,
            current_revision=OTHER_REVISION,
        )
    elif mutation == "source_revision":
        case.catalog.sync = replace(
            case.catalog.sync,
            source_revision="schema_version:2",
        )
    elif mutation == "payload_count":
        payload = {**case.payload, "returned_rows": 9}
        case = _case()
        operation = _snapshot(
            payload=payload,
            evidence_kind="data.sqlite.query_result",
            content_hash="sha256:"
            + sha256(canonical_json(payload).encode("utf-8")).hexdigest(),
            blob_id=None,
        )
        case.reader._operation_store = cast(  # noqa: SLF001
            OperationStore,
            OperationStoreStub(operation),
        )
    elif mutation == "payload_hash":
        evidence = case.operation.snapshot.evidence[0]
        forged = replace(evidence, content_hash="sha256:" + "f" * 64)
        snapshot = replace(case.operation.snapshot, evidence=(forged,))
        case.reader._operation_store = cast(  # noqa: SLF001
            OperationStore,
            OperationStoreStub(VersionedOperation(snapshot=snapshot, revision=1)),
        )

    with pytest.raises(DataEvidenceDatasetError) as error:
        await case.reader.load_dataset(
            operation_id="operation-1",
            evidence_id=case.evidence_id,
        )
    assert error.value.code == expected_code


async def test_blob_content_and_metadata_are_verified_before_use() -> None:
    artifact = canonical_json([{"id": 1, "name": "Ada"}]).encode("utf-8")
    corrupt = artifact[:-1] + b" "
    case = _case(file_evidence=True, blob_content=artifact, opened_blob_content=corrupt)

    with pytest.raises(DataEvidenceDatasetError) as integrity_error:
        await case.reader.load_dataset(
            operation_id="operation-1",
            evidence_id=case.evidence_id,
        )
    assert integrity_error.value.code == "blob_integrity_failed"

    case = _case(file_evidence=True, blob_content=artifact)
    assert case.blob._metadata is not None
    case.blob._metadata = replace(case.blob._metadata, evidence_id="evidence-other")
    with pytest.raises(DataEvidenceDatasetError) as metadata_error:
        await case.reader.load_dataset(
            operation_id="operation-1",
            evidence_id=case.evidence_id,
        )
    assert metadata_error.value.code == "blob_metadata_invalid"


async def test_blob_rows_must_match_inline_prefix_and_declared_columns() -> None:
    mismatched = canonical_json([{"id": 2, "name": "Grace"}]).encode("utf-8")
    case = _case(file_evidence=True, blob_content=mismatched)
    with pytest.raises(DataEvidenceDatasetError) as mismatch_error:
        await case.reader.load_dataset(
            operation_id="operation-1",
            evidence_id=case.evidence_id,
        )
    assert mismatch_error.value.code == "blob_payload_mismatch"

    unknown_column = canonical_json(
        [{"id": 1, "name": "Ada", "secret": "unexpected"}]
    ).encode("utf-8")
    case = _case(file_evidence=True, blob_content=unknown_column)
    with pytest.raises(DataEvidenceDatasetError) as columns_error:
        await case.reader.load_dataset(
            operation_id="operation-1",
            evidence_id=case.evidence_id,
        )
    assert columns_error.value.code == "blob_payload_invalid"


async def test_requested_operation_and_evidence_scope_cannot_escape() -> None:
    case = _case()
    with pytest.raises(DataEvidenceDatasetError) as operation_error:
        await case.reader.load_dataset(
            operation_id="operation-other",
            evidence_id=case.evidence_id,
        )
    assert operation_error.value.code == "operation_scope_mismatch"

    with pytest.raises(DataEvidenceDatasetError) as evidence_error:
        await case.reader.load_dataset(
            operation_id="operation-1",
            evidence_id="evidence-other",
        )
    assert evidence_error.value.code == "evidence_not_found"
