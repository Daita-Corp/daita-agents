from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any

import pytest

import daita.capabilities as capability_models
from daita.capabilities import (
    AccessMode,
    Capability,
    CapabilityExecutionError,
    CapabilityRegistry,
    EvidenceCandidate,
    EvidenceValidationError,
    ExecutionRequest,
    RiskLevel,
    ToolView,
)
from daita.llm.models import (
    CanonicalMessage,
    FinishReason,
    MessageRole,
    ModelRequest,
    ModelResponse,
    TextBlock,
    ToolCall,
)
from daita.loop.models import LoopPhase
from daita.operations.checkpoints import OperationSnapshot
from daita.operations.leases import TaskLeaseGuard
from daita.operations.models import (
    ActionProposal,
    AgentTrigger,
    TaskStatus,
    TriggerKind,
)
from daita.operations.runtime import OperationRuntime, OperationStateError
from daita.operations.store import (
    CommitResult,
    InMemoryOperationStore,
    InvalidOperationCheckpointError,
    StaleTaskFenceError,
    VersionedOperation,
)
from daita.storage.blobs import (
    BlobMetadata,
    BlobPut,
    BlobStoreError,
    LocalBlobStore,
)

NOW = datetime(2026, 7, 18, 12, 0, tzinfo=timezone.utc)
ARTIFACT_CONTENT = b'{"rows":[{"key":"alpha","value":"ALPHA"}]}'
ARTIFACT_DIGEST = "sha256:" + hashlib.sha256(ARTIFACT_CONTENT).hexdigest()
ARTIFACT_ENCRYPTION = {
    "algorithm": "AES-256-GCM",
    "key_id": "agent-home-key-1",
}


class DeterministicIds:
    def __init__(self) -> None:
        self._counts: defaultdict[str, int] = defaultdict(int)

    def __call__(self, prefix: str) -> str:
        self._counts[prefix] += 1
        return f"{prefix}-{self._counts[prefix]}"


class CandidateExecutor:
    def __init__(self, candidate: EvidenceCandidate) -> None:
        self.executor_id = "fake.read.executor"
        self.candidate = candidate
        self.requests: list[ExecutionRequest] = []

    async def execute(self, request: ExecutionRequest) -> EvidenceCandidate:
        self.requests.append(request)
        return self.candidate


class RecordingLocalBlobStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.delegate = LocalBlobStore(root)
        self.requests: list[tuple[BlobPut, bytes]] = []
        self.committed: dict[str, BlobMetadata] = {}

    async def put(self, request: BlobPut, content: bytes) -> BlobMetadata:
        self.requests.append((request, content))
        metadata = await self.delegate.put(request, content)
        self.committed[request.blob_id] = metadata
        return metadata


class DurableBarrierBlobStore(RecordingLocalBlobStore):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.put_committed = asyncio.Event()
        self.release_put = asyncio.Event()

    async def put(self, request: BlobPut, content: bytes) -> BlobMetadata:
        metadata = await super().put(request, content)
        self.put_committed.set()
        await self.release_put.wait()
        return metadata


class FailingBlobStore:
    def __init__(self) -> None:
        self.requests: list[tuple[BlobPut, bytes]] = []

    async def put(self, request: BlobPut, content: bytes) -> BlobMetadata:
        self.requests.append((request, content))
        raise BlobStoreError("injected durable blob put failure")


class MismatchedMetadataBlobStore(RecordingLocalBlobStore):
    async def put(self, request: BlobPut, content: bytes) -> BlobMetadata:
        metadata = await super().put(request, content)
        return replace(metadata, evidence_id="evidence-forged")


class RejectingTerminalFencedStore(InMemoryOperationStore):
    def __init__(self) -> None:
        super().__init__(clock=lambda: NOW)
        self.fenced_calls = 0
        self.before_rejection: VersionedOperation | None = None

    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        self.fenced_calls += 1
        if self.fenced_calls == 2:
            self.before_rejection = await self.load(guard.operation_id)
            raise InvalidOperationCheckpointError(
                guard.operation_id,
                "injected terminal fenced evidence failure",
            )
        return await super().commit_fenced(
            snapshot,
            expected_revision=expected_revision,
            guard=guard,
        )


class StaleTerminalFenceStore(InMemoryOperationStore):
    def __init__(self) -> None:
        super().__init__(clock=lambda: NOW)
        self.fenced_calls = 0
        self.before_rejection: VersionedOperation | None = None

    async def commit_fenced(
        self,
        snapshot: OperationSnapshot,
        *,
        expected_revision: int,
        guard: TaskLeaseGuard,
    ) -> CommitResult:
        self.fenced_calls += 1
        if self.fenced_calls == 2:
            self.before_rejection = await self.load(guard.operation_id)
            raise StaleTaskFenceError(
                guard.operation_id,
                guard.task_id,
                guard.fencing_token,
                guard.fencing_token + 1,
            )
        return await super().commit_fenced(
            snapshot,
            expected_revision=expected_revision,
            guard=guard,
        )


@dataclass(frozen=True)
class RuntimeCase:
    runtime: OperationRuntime
    executor: CandidateExecutor
    store: InMemoryOperationStore
    operation_id: str
    turn_id: str


def _artifact() -> Any:
    artifact_type = getattr(capability_models, "EvidenceArtifact", None)
    assert artifact_type is not None, (
        "daita.capabilities must define EvidenceArtifact before the runtime can "
        "materialize blob-backed evidence"
    )
    return artifact_type(
        content=ARTIFACT_CONTENT,
        media_type="application/json",
        sensitivity_class="internal",
        retention_class="operation",
        encryption_metadata=ARTIFACT_ENCRYPTION,
    )


def _candidate(
    *,
    payload: Mapping[str, object] | None = None,
) -> EvidenceCandidate:
    candidate_type: Any = EvidenceCandidate
    return candidate_type(
        kind="fake.read.result",
        schema_version=1,
        payload=({"key": "alpha", "value": "ALPHA"} if payload is None else payload),
        artifact=_artifact(),
    )


def _capability() -> Capability:
    return Capability(
        id="fake.read",
        owner="blob-evidence-tests",
        description="Read a value and return one durable artifact.",
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
            "additionalProperties": False,
        },
        output_evidence_kind="fake.read.result",
        output_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
            "additionalProperties": False,
        },
        output_schema_version=1,
        executor_id="fake.read.executor",
        access_mode=AccessMode.READ,
        risk=RiskLevel.LOW,
        side_effecting=False,
        idempotent=True,
        replay_safe=True,
    )


async def _runtime_case(
    candidate: EvidenceCandidate,
    *,
    blob_store: object,
    store: InMemoryOperationStore | None = None,
) -> RuntimeCase:
    executor = CandidateExecutor(candidate)
    capability = _capability()
    registry = CapabilityRegistry(
        capabilities=(capability,),
        executors=(executor,),
        tool_views=(
            ToolView(
                name="read_fake",
                capability_id=capability.id,
                description="Read one fake value.",
            ),
        ),
    )
    operation_store = store or InMemoryOperationStore(clock=lambda: NOW)
    runtime_type: Any = OperationRuntime
    runtime = runtime_type(
        clock=lambda: NOW,
        id_factory=DeterministicIds(),
        capabilities=registry,
        store=operation_store,
        blob_store=blob_store,
    )
    started = await runtime.begin(
        AgentTrigger(
            id="trigger-1",
            agent_id="agent-1",
            kind=TriggerKind.USER,
            source_id="user-1",
            payload={"message": "read alpha"},
            created_at=NOW,
        )
    )
    turn = await runtime.begin_turn(started.operation.id)
    request = ModelRequest(
        operation_id=started.operation.id,
        turn_id=turn.id,
        messages=(
            CanonicalMessage(
                agent_id="agent-1",
                operation_id=started.operation.id,
                turn_id=turn.id,
                role=MessageRole.USER,
                content=(TextBlock("Read alpha."),),
            ),
        ),
        tools=registry.tool_definitions(),
    )
    model_call = await runtime.begin_model_call(
        started.operation.id,
        turn.id,
        "mock:scripted",
        request,
    )
    await runtime.record_model_response(
        started.operation.id,
        model_call.id,
        ModelResponse(
            finish_reason=FinishReason.TOOL_CALLS,
            tool_calls=(
                ToolCall(
                    id="call-1",
                    name="read_fake",
                    arguments={"key": "alpha"},
                ),
            ),
        ),
        next_phase=LoopPhase.VALIDATING_ACTION,
    )
    return RuntimeCase(
        runtime=runtime,
        executor=executor,
        store=operation_store,
        operation_id=started.operation.id,
        turn_id=turn.id,
    )


def _proposal(case: RuntimeCase) -> ActionProposal:
    return ActionProposal(
        operation_id=case.operation_id,
        turn_id=case.turn_id,
        call_id="call-1",
        capability_id="fake.read",
        arguments={"key": "alpha"},
        proposed_at=NOW,
    )


async def _assert_durable_blob(
    root: Path,
    request: BlobPut,
    expected_content: bytes,
) -> BlobMetadata:
    reopened = LocalBlobStore(root)
    metadata = await reopened.metadata(request.blob_id)
    assert metadata is not None
    async with await reopened.open(request.blob_id) as reader:
        assert reader.metadata == metadata
        assert await reader.read(len(expected_content) + 1) == expected_content
    return metadata


def _assert_no_terminal_evidence(snapshot: OperationSnapshot) -> None:
    assert snapshot.evidence == ()
    assert snapshot.tasks[0].evidence_ids == ()
    assert snapshot.tasks[0].status is TaskStatus.RUNNING
    assert snapshot.task_leases[0].released_at is None
    terminal_types = {"executor.completed", "evidence.accepted", "task.succeeded"}
    assert not terminal_types.intersection(event.type for event in snapshot.events)


async def test_invalid_candidate_is_rejected_before_any_blob_put(
    tmp_path: Path,
) -> None:
    blob_store = RecordingLocalBlobStore(tmp_path / "blobs")
    case = await _runtime_case(
        _candidate(payload={"key": "alpha"}),
        blob_store=blob_store,
    )

    with pytest.raises(EvidenceValidationError, match="required field"):
        await case.runtime.submit(_proposal(case))

    snapshot = await case.runtime.inspect(case.operation_id)
    assert len(case.executor.requests) == 1
    assert blob_store.requests == []
    assert snapshot.evidence == ()
    assert snapshot.tasks[0].evidence_ids == ()
    assert not any(event.type == "evidence.accepted" for event in snapshot.events)


async def test_valid_artifact_uses_runtime_owned_ids_and_exact_blob_provenance(
    tmp_path: Path,
) -> None:
    blob_store = RecordingLocalBlobStore(tmp_path / "blobs")
    candidate = _candidate()
    case = await _runtime_case(candidate, blob_store=blob_store)

    evidence = await case.runtime.submit(_proposal(case))

    snapshot = await case.runtime.inspect(case.operation_id)
    task = snapshot.tasks[0]
    assert len(blob_store.requests) == 1
    request, content = blob_store.requests[0]
    assert content == ARTIFACT_CONTENT
    assert request == BlobPut(
        blob_id="blob-1",
        media_type="application/json",
        created_at=NOW,
        sensitivity_class="internal",
        retention_class="operation",
        operation_id=case.operation_id,
        task_id=task.id,
        evidence_id=evidence.id,
        expected_digest=ARTIFACT_DIGEST,
        encryption_metadata=ARTIFACT_ENCRYPTION,
    )
    assert evidence.id == "evidence-1"
    assert evidence.blob_id == request.blob_id
    assert evidence.content_hash == ARTIFACT_DIGEST
    assert task.evidence_ids == (evidence.id,)
    assert snapshot.evidence == (evidence,)
    assert not hasattr(candidate.artifact, "blob_id")
    assert not hasattr(candidate.artifact, "evidence_id")

    metadata = await _assert_durable_blob(
        blob_store.root,
        request,
        ARTIFACT_CONTENT,
    )
    assert metadata == blob_store.committed[request.blob_id]
    assert metadata.digest == evidence.content_hash
    assert metadata.operation_id == evidence.operation_id
    assert metadata.task_id == evidence.task_id
    assert metadata.evidence_id == evidence.id


async def test_blob_is_durable_before_fenced_evidence_acceptance(
    tmp_path: Path,
) -> None:
    operation_store = InMemoryOperationStore(clock=lambda: NOW)
    blob_store = DurableBarrierBlobStore(tmp_path / "blobs")
    case = await _runtime_case(
        _candidate(),
        blob_store=blob_store,
        store=operation_store,
    )

    submission = asyncio.create_task(case.runtime.submit(_proposal(case)))
    try:
        await asyncio.wait_for(blob_store.put_committed.wait(), timeout=1.0)
        assert len(blob_store.requests) == 1
        request, content = blob_store.requests[0]
        durable = await _assert_durable_blob(blob_store.root, request, content)
        assert durable == blob_store.committed[request.blob_id]

        before_acceptance = await operation_store.load(case.operation_id)
        _assert_no_terminal_evidence(before_acceptance.snapshot)
        assert before_acceptance.snapshot.events[-1].type == "executor.started"
    finally:
        blob_store.release_put.set()

    evidence = await submission
    accepted = await operation_store.load(case.operation_id)
    assert accepted.snapshot.evidence == (evidence,)
    assert evidence.blob_id == request.blob_id
    assert evidence.content_hash == durable.digest


async def test_blob_put_failure_cannot_publish_evidence(
    tmp_path: Path,
) -> None:
    del tmp_path
    blob_store = FailingBlobStore()
    case = await _runtime_case(_candidate(), blob_store=blob_store)

    with pytest.raises((BlobStoreError, CapabilityExecutionError)):
        await case.runtime.submit(_proposal(case))

    snapshot = await case.runtime.inspect(case.operation_id)
    assert len(blob_store.requests) == 1
    assert snapshot.evidence == ()
    assert snapshot.tasks[0].evidence_ids == ()
    assert snapshot.tasks[0].status is not TaskStatus.SUCCEEDED
    assert not any(
        event.type in {"evidence.accepted", "task.succeeded"}
        for event in snapshot.events
    )


async def test_artifact_without_a_blob_store_fails_before_evidence_acceptance(
    tmp_path: Path,
) -> None:
    del tmp_path
    case = await _runtime_case(_candidate(), blob_store=None)

    with pytest.raises(CapabilityExecutionError, match="artifact persistence"):
        await case.runtime.submit(_proposal(case))

    snapshot = await case.runtime.inspect(case.operation_id)
    assert snapshot.evidence == ()
    assert snapshot.tasks[0].status is TaskStatus.FAILED
    assert snapshot.tasks[0].error_code == "evidence_blob_failed"
    assert not any(event.type == "evidence.accepted" for event in snapshot.events)


async def test_mismatched_blob_metadata_is_rejected_after_durable_put(
    tmp_path: Path,
) -> None:
    blob_store = MismatchedMetadataBlobStore(tmp_path / "blobs")
    case = await _runtime_case(_candidate(), blob_store=blob_store)

    with pytest.raises(CapabilityExecutionError, match="artifact persistence"):
        await case.runtime.submit(_proposal(case))

    request, content = blob_store.requests[0]
    orphan = await _assert_durable_blob(blob_store.root, request, content)
    assert orphan.evidence_id == request.evidence_id
    snapshot = await case.runtime.inspect(case.operation_id)
    assert snapshot.evidence == ()
    assert snapshot.tasks[0].status is TaskStatus.FAILED
    assert snapshot.tasks[0].error_code == "evidence_blob_failed"


async def test_cancellation_after_durable_put_cannot_accept_evidence(
    tmp_path: Path,
) -> None:
    blob_store = DurableBarrierBlobStore(tmp_path / "blobs")
    case = await _runtime_case(_candidate(), blob_store=blob_store)
    submission = asyncio.create_task(case.runtime.submit(_proposal(case)))

    await asyncio.wait_for(blob_store.put_committed.wait(), timeout=1.0)
    submission.cancel()
    with pytest.raises(asyncio.CancelledError):
        await submission

    request, content = blob_store.requests[0]
    await _assert_durable_blob(blob_store.root, request, content)
    snapshot = await case.runtime.inspect(case.operation_id)
    _assert_no_terminal_evidence(snapshot)


async def test_terminal_fenced_failure_after_put_leaves_a_durable_orphan(
    tmp_path: Path,
) -> None:
    operation_store = RejectingTerminalFencedStore()
    blob_store = RecordingLocalBlobStore(tmp_path / "blobs")
    case = await _runtime_case(
        _candidate(),
        blob_store=blob_store,
        store=operation_store,
    )

    with pytest.raises(OperationStateError, match="fenced") as caught:
        await case.runtime.submit(_proposal(case))

    assert isinstance(caught.value.__cause__, InvalidOperationCheckpointError)
    assert len(blob_store.requests) == 1
    request, content = blob_store.requests[0]
    orphan = await _assert_durable_blob(blob_store.root, request, content)
    assert orphan.evidence_id == request.evidence_id

    assert operation_store.before_rejection is not None
    after_rejection = await operation_store.load(case.operation_id)
    assert after_rejection == operation_store.before_rejection
    _assert_no_terminal_evidence(after_rejection.snapshot)


async def test_stale_fence_after_put_cannot_accept_blob_evidence(
    tmp_path: Path,
) -> None:
    operation_store = StaleTerminalFenceStore()
    blob_store = RecordingLocalBlobStore(tmp_path / "blobs")
    case = await _runtime_case(
        _candidate(),
        blob_store=blob_store,
        store=operation_store,
    )

    with pytest.raises(OperationStateError, match="fenced") as caught:
        await case.runtime.submit(_proposal(case))

    assert isinstance(caught.value.__cause__, StaleTaskFenceError)
    assert len(blob_store.requests) == 1
    request, content = blob_store.requests[0]
    orphan = await _assert_durable_blob(blob_store.root, request, content)
    assert orphan.evidence_id == request.evidence_id

    assert operation_store.before_rejection is not None
    after_rejection = await operation_store.load(case.operation_id)
    assert after_rejection == operation_store.before_rejection
    _assert_no_terminal_evidence(after_rejection.snapshot)
