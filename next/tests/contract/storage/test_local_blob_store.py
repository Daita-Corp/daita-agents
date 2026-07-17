from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
from pathlib import Path
from typing import Any

import pytest

from daita.storage.blobs import (
    BlobIdentityConflictError,
    BlobIntegrityError,
    BlobMetadata,
    BlobPut,
    BlobRevisionConflict,
    BlobStoreError,
    BlobUnavailableError,
    LocalBlobStore,
)

NOW = datetime(2026, 7, 18, 10, 11, 12, 123_456, tzinfo=timezone.utc)
TOMBSTONED_AT = NOW + timedelta(seconds=1)
DELETED_AT = NOW + timedelta(seconds=2)


def _digest(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def _object_path(root: Path, content: bytes) -> Path:
    digest_hex = hashlib.sha256(content).hexdigest()
    return root / "sha256" / digest_hex[:2] / digest_hex


def _request(
    blob_id: str,
    content: bytes,
    *,
    operation_id: str = "operation-blob",
    task_id: str = "task-blob",
    evidence_id: str = "evidence-blob",
) -> BlobPut:
    return BlobPut(
        blob_id=blob_id,
        media_type="application/octet-stream",
        created_at=NOW,
        sensitivity_class="internal",
        retention_class="operation",
        operation_id=operation_id,
        task_id=task_id,
        evidence_id=evidence_id,
        expected_digest=_digest(content),
        encryption_metadata={"scheme": None, "labels": ["local", {"v": 1}]},
    )


def _assert_initial_metadata(
    metadata: BlobMetadata,
    request: BlobPut,
    content: bytes,
) -> None:
    assert metadata.blob_id == request.blob_id
    assert metadata.digest == _digest(content)
    assert metadata.size_bytes == len(content)
    assert metadata.media_type == request.media_type
    assert metadata.created_at == request.created_at
    assert metadata.sensitivity_class == request.sensitivity_class
    assert metadata.retention_class == request.retention_class
    assert metadata.operation_id == request.operation_id
    assert metadata.task_id == request.task_id
    assert metadata.evidence_id == request.evidence_id
    assert metadata.encryption_metadata == request.encryption_metadata
    assert metadata.version == 1
    assert metadata.tombstoned_at is None
    assert metadata.deleted_at is None


async def test_put_open_and_reopen_use_exact_content_addressed_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"bounded content-addressed blob reads"
    request = _request("blob-roundtrip", content)
    store = LocalBlobStore(root, max_blob_bytes=1024)

    committed = await store.put(request, content)

    _assert_initial_metadata(committed, request, content)
    object_path = _object_path(root, content)
    assert object_path.read_bytes() == content
    assert sorted(path for path in (root / "sha256").rglob("*") if path.is_file()) == [
        object_path
    ]
    assert not hasattr(committed, "path")

    reader = await store.open(request.blob_id)
    assert reader.metadata == committed
    assert not hasattr(reader, "path")
    async with reader as opened:
        assert opened is reader
        assert await opened.read(0) == b""
        assert await opened.read(7) == content[:7]
        assert await opened.read(5) == content[7:12]
        assert await opened.read(10_000) == content[12:]
        assert await opened.read(1) == b""
        with pytest.raises(ValueError, match="size|negative|non-negative"):
            await opened.read(-1)
    with pytest.raises(ValueError, match="closed"):
        await reader.read(1)

    reopened = LocalBlobStore(root)
    assert await reopened.metadata(request.blob_id) == committed
    async with await reopened.open(request.blob_id) as reopened_reader:
        assert reopened_reader.metadata == committed
        assert await reopened_reader.read(len(content)) == content
        assert await reopened_reader.read(1) == b""


async def test_put_rejects_digest_mismatch_and_configured_size_limit(
    tmp_path: Path,
) -> None:
    content = b"five"
    mismatch_root = tmp_path / "mismatch"
    request = replace(
        _request("blob-mismatch", content),
        expected_digest="sha256:" + "0" * 64,
    )
    mismatch_store = LocalBlobStore(mismatch_root)

    with pytest.raises(BlobIntegrityError) as mismatch:
        await mismatch_store.put(request, content)

    assert mismatch.value.blob_id == request.blob_id
    assert await mismatch_store.metadata(request.blob_id) is None
    assert not _object_path(mismatch_root, content).exists()

    limit_root = tmp_path / "limit"
    limit_store = LocalBlobStore(limit_root, max_blob_bytes=len(content) - 1)
    limit_request = _request("blob-too-large", content)

    with pytest.raises(BlobStoreError, match="size|large|maximum|limit"):
        await limit_store.put(limit_request, content)

    assert await limit_store.metadata(limit_request.blob_id) is None
    assert not _object_path(limit_root, content).exists()


async def test_same_blob_id_identical_retry_is_idempotent_without_version_bump(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"one logical blob identity"
    request = _request("blob-idempotent", content)
    first_store = LocalBlobStore(root)

    first = await first_store.put(request, content)
    retried = await first_store.put(request, content)
    reopened_retry = await LocalBlobStore(root).put(request, content)

    assert retried == first
    assert reopened_retry == first
    assert first.version == 1
    assert await LocalBlobStore(root).metadata(request.blob_id) == first
    assert _object_path(root, content).read_bytes() == content


@pytest.mark.parametrize(
    "changes",
    (
        {"media_type": "text/plain"},
        {"sensitivity_class": "restricted"},
        {"retention_class": "legal-hold"},
        {"operation_id": "operation-other"},
        {"encryption_metadata": {"scheme": "changed"}},
    ),
)
async def test_same_blob_id_rejects_changed_immutable_metadata(
    tmp_path: Path,
    changes: dict[str, Any],
) -> None:
    root = tmp_path / "blobs"
    content = b"immutable logical metadata"
    original_request = _request("blob-immutable", content)
    store = LocalBlobStore(root)
    original = await store.put(original_request, content)

    with pytest.raises(BlobIdentityConflictError) as conflict:
        await store.put(replace(original_request, **changes), content)

    assert conflict.value.blob_id == original.blob_id
    assert await store.metadata(original.blob_id) == original
    assert _object_path(root, content).read_bytes() == content


async def test_same_blob_id_rejects_changed_content_without_storing_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    original_content = b"original immutable content"
    changed_content = b"changed immutable content"
    store = LocalBlobStore(root)
    original_request = _request("blob-content-conflict", original_content)
    original = await store.put(original_request, original_content)

    with pytest.raises(BlobIdentityConflictError) as conflict:
        await store.put(
            _request(original_request.blob_id, changed_content),
            changed_content,
        )

    assert conflict.value.blob_id == original.blob_id
    assert await store.metadata(original.blob_id) == original
    assert _object_path(root, original_content).read_bytes() == original_content
    assert not _object_path(root, changed_content).exists()


async def test_two_instances_deduplicate_concurrent_content_with_distinct_provenance(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"shared physical bytes with distinct logical producers"
    first_request = _request(
        "blob-shared-a",
        content,
        operation_id="operation-a",
        task_id="task-a",
        evidence_id="evidence-a",
    )
    second_request = _request(
        "blob-shared-b",
        content,
        operation_id="operation-b",
        task_id="task-b",
        evidence_id="evidence-b",
    )
    first_store = LocalBlobStore(root)
    second_store = LocalBlobStore(root)

    first, second = await asyncio.gather(
        first_store.put(first_request, content),
        second_store.put(second_request, content),
    )

    assert first.digest == second.digest == _digest(content)
    assert first.blob_id != second.blob_id
    assert (first.operation_id, first.task_id, first.evidence_id) == (
        "operation-a",
        "task-a",
        "evidence-a",
    )
    assert (second.operation_id, second.task_id, second.evidence_id) == (
        "operation-b",
        "task-b",
        "evidence-b",
    )
    object_path = _object_path(root, content)
    assert object_path.read_bytes() == content
    assert sorted(path for path in (root / "sha256").rglob("*") if path.is_file()) == [
        object_path
    ]
    reopened = LocalBlobStore(root)
    assert await reopened.metadata(first.blob_id) == first
    assert await reopened.metadata(second.blob_id) == second


async def test_tombstone_and_delete_use_expected_version_and_preserve_metadata(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"terminal blob lifecycle"
    request = _request("blob-lifecycle", content)
    store = LocalBlobStore(root)
    initial = await store.put(request, content)

    with pytest.raises(BlobRevisionConflict) as stale_tombstone:
        await store.tombstone(
            request.blob_id,
            expected_version=2,
            at=TOMBSTONED_AT,
        )
    assert stale_tombstone.value.expected_version == 2
    assert stale_tombstone.value.actual_version == 1
    assert await store.metadata(request.blob_id) == initial

    tombstoned = await store.tombstone(
        request.blob_id,
        expected_version=1,
        at=TOMBSTONED_AT,
    )
    assert tombstoned == replace(
        initial,
        version=2,
        tombstoned_at=TOMBSTONED_AT,
    )
    with pytest.raises(BlobUnavailableError) as unavailable:
        await store.open(request.blob_id)
    assert unavailable.value.state == "tombstoned"

    with pytest.raises(BlobRevisionConflict) as stale_delete:
        await LocalBlobStore(root).delete(
            request.blob_id,
            expected_version=1,
            at=DELETED_AT,
        )
    assert stale_delete.value.actual_version == 2
    assert await store.metadata(request.blob_id) == tombstoned

    deleted = await LocalBlobStore(root).delete(
        request.blob_id,
        expected_version=2,
        at=DELETED_AT,
    )
    assert deleted == replace(tombstoned, version=3, deleted_at=DELETED_AT)
    assert await LocalBlobStore(root).metadata(request.blob_id) == deleted
    with pytest.raises(BlobUnavailableError) as deleted_unavailable:
        await store.open(request.blob_id)
    assert deleted_unavailable.value.state == "deleted"
    assert not _object_path(root, content).exists()


async def test_cross_instance_lifecycle_cas_allows_exactly_one_winner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"one lifecycle transition winner"
    request = _request("blob-cas", content)
    await LocalBlobStore(root).put(request, content)
    first_store = LocalBlobStore(root)
    second_store = LocalBlobStore(root)

    outcomes = await asyncio.gather(
        first_store.tombstone(
            request.blob_id,
            expected_version=1,
            at=TOMBSTONED_AT,
        ),
        second_store.tombstone(
            request.blob_id,
            expected_version=1,
            at=TOMBSTONED_AT + timedelta(microseconds=1),
        ),
        return_exceptions=True,
    )

    winners = [outcome for outcome in outcomes if isinstance(outcome, BlobMetadata)]
    losers = [
        outcome for outcome in outcomes if isinstance(outcome, BlobRevisionConflict)
    ]
    assert len(winners) == 1
    assert len(losers) == 1
    assert winners[0].version == 2
    assert losers[0].expected_version == 1
    assert losers[0].actual_version == 2
    assert await LocalBlobStore(root).metadata(request.blob_id) == winners[0]


async def test_shared_content_is_removed_only_after_last_logical_record_is_deleted(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"reference-counted content address"
    first_request = _request("blob-retain-a", content)
    second_request = _request(
        "blob-retain-b",
        content,
        operation_id="operation-b",
        task_id="task-b",
        evidence_id="evidence-b",
    )
    store = LocalBlobStore(root)
    first = await store.put(first_request, content)
    second = await store.put(second_request, content)
    object_path = _object_path(root, content)

    first_tombstone = await store.tombstone(
        first.blob_id,
        expected_version=1,
        at=TOMBSTONED_AT,
    )
    first_deleted = await store.delete(
        first.blob_id,
        expected_version=2,
        at=DELETED_AT,
    )

    assert first_deleted.tombstoned_at == first_tombstone.tombstoned_at
    assert first_deleted.deleted_at == DELETED_AT
    assert object_path.read_bytes() == content
    async with await LocalBlobStore(root).open(second.blob_id) as reader:
        assert await reader.read(len(content)) == content

    second_tombstone = await LocalBlobStore(root).tombstone(
        second.blob_id,
        expected_version=1,
        at=TOMBSTONED_AT,
    )
    second_deleted = await LocalBlobStore(root).delete(
        second.blob_id,
        expected_version=second_tombstone.version,
        at=DELETED_AT,
    )

    assert not object_path.exists()
    reopened = LocalBlobStore(root)
    assert await reopened.metadata(first.blob_id) == first_deleted
    assert await reopened.metadata(second.blob_id) == second_deleted


@pytest.mark.parametrize("damage", ("corrupt", "symlink"))
async def test_open_fails_closed_for_corrupt_or_symlinked_content(
    tmp_path: Path,
    damage: str,
) -> None:
    root = tmp_path / "blobs"
    content = b"content that must be verified before exposure"
    request = _request(f"blob-object-{damage}", content)
    committed = await LocalBlobStore(root).put(request, content)
    object_path = _object_path(root, content)

    if damage == "corrupt":
        object_path.write_bytes(b"x" * len(content))
    else:
        outside = tmp_path / "outside-object.bin"
        outside.write_bytes(content)
        object_path.unlink()
        object_path.symlink_to(outside)

    with pytest.raises(BlobIntegrityError) as integrity:
        await LocalBlobStore(root).open(request.blob_id)

    assert integrity.value.blob_id == request.blob_id
    assert integrity.value.digest == committed.digest


async def test_corrupt_private_logical_record_fails_closed_without_layout_contract(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"content with independently durable logical metadata"
    request = _request("blob-corrupt-record", content)
    await LocalBlobStore(root).put(request, content)
    object_root = root / "sha256"
    private_record_files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_relative_to(object_root)
    )
    assert private_record_files

    for path in private_record_files:
        path.write_bytes(b"not a valid durable blob record\n")

    reopened = LocalBlobStore(root)
    with pytest.raises(BlobStoreError):
        await reopened.metadata(request.blob_id)
    with pytest.raises(BlobStoreError):
        await reopened.open(request.blob_id)
