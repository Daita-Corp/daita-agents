from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from daita.storage import blobs as blob_owner
from daita.storage.blobs import (
    BlobIdentityConflictError,
    BlobIntegrityError,
    BlobMetadata,
    BlobPut,
    BlobRetentionError,
    BlobStoreError,
    BlobUnavailableError,
    LocalBlobStore,
)

NOW = datetime(2026, 7, 18, 14, 30, 15, 654_321, tzinfo=timezone.utc)
TOMBSTONED_AT = NOW + timedelta(seconds=1)
DELETED_AT = NOW + timedelta(seconds=2)


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _request(blob_id: str, content: bytes) -> BlobPut:
    return BlobPut(
        blob_id=blob_id,
        media_type="application/octet-stream",
        created_at=NOW,
        sensitivity_class="internal",
        retention_class="operation",
        operation_id="operation-adversarial",
        task_id="task-adversarial",
        evidence_id="evidence-adversarial",
        expected_digest=_digest(content),
        encryption_metadata={"scheme": None},
    )


def _object_path(root: Path, content: bytes) -> Path:
    digest_hex = hashlib.sha256(content).hexdigest()
    return root / "sha256" / digest_hex[:2] / digest_hex


def _only_private_manifest(root: Path) -> Path:
    candidates = tuple(
        path for path in root.rglob("*.json") if path.is_file() or path.is_symlink()
    )
    assert len(candidates) == 1
    return candidates[0]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _checksummed_envelope(record: dict[str, Any]) -> dict[str, Any]:
    checksum = hashlib.sha256(_canonical_json(record).encode("utf-8")).hexdigest()
    return {"checksum": "sha256:" + checksum, "record": record}


async def test_concurrent_conflicting_same_identity_puts_have_one_winner(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    first_content = b"first contender"
    second_content = b"second contender"
    blob_id = "blob-contended"

    outcomes = await asyncio.gather(
        LocalBlobStore(root).put(_request(blob_id, first_content), first_content),
        LocalBlobStore(root).put(_request(blob_id, second_content), second_content),
        return_exceptions=True,
    )

    winners = [outcome for outcome in outcomes if isinstance(outcome, BlobMetadata)]
    conflicts = [
        outcome
        for outcome in outcomes
        if isinstance(outcome, BlobIdentityConflictError)
    ]
    assert len(winners) == 1
    assert len(conflicts) == 1
    assert conflicts[0].blob_id == blob_id

    winner = winners[0]
    expected_content = (
        first_content if winner.digest == _digest(first_content) else second_content
    )
    losing_content = (
        second_content if expected_content == first_content else first_content
    )
    assert await LocalBlobStore(root).metadata(blob_id) == winner
    assert _object_path(root, expected_content).read_bytes() == expected_content
    assert not _object_path(root, losing_content).exists()


async def test_put_cannot_resurrect_tombstoned_or_deleted_identity(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"terminal identities cannot be resurrected"
    request = _request("blob-terminal-put", content)
    store = LocalBlobStore(root)
    initial = await store.put(request, content)
    tombstoned = await store.tombstone(
        request.blob_id,
        expected_version=initial.version,
        at=TOMBSTONED_AT,
    )

    with pytest.raises(BlobIdentityConflictError):
        await LocalBlobStore(root).put(request, content)
    assert await store.metadata(request.blob_id) == tombstoned

    deleted = await store.delete(
        request.blob_id,
        expected_version=tombstoned.version,
        at=DELETED_AT,
    )
    with pytest.raises(BlobIdentityConflictError):
        await LocalBlobStore(root).put(request, content)
    assert await LocalBlobStore(root).metadata(request.blob_id) == deleted
    assert not _object_path(root, content).exists()


async def test_invalid_and_repeated_lifecycle_transitions_are_non_mutating(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"one-way lifecycle"
    request = _request("blob-one-way", content)
    store = LocalBlobStore(root)
    initial = await store.put(request, content)

    with pytest.raises(BlobRetentionError):
        await store.delete(
            request.blob_id,
            expected_version=initial.version,
            at=DELETED_AT,
        )
    assert await store.metadata(request.blob_id) == initial

    tombstoned = await store.tombstone(
        request.blob_id,
        expected_version=initial.version,
        at=TOMBSTONED_AT,
    )
    with pytest.raises(BlobUnavailableError) as repeated_tombstone:
        await store.tombstone(
            request.blob_id,
            expected_version=tombstoned.version,
            at=TOMBSTONED_AT + timedelta(microseconds=1),
        )
    assert repeated_tombstone.value.state == "tombstoned"
    assert await store.metadata(request.blob_id) == tombstoned

    deleted = await store.delete(
        request.blob_id,
        expected_version=tombstoned.version,
        at=DELETED_AT,
    )
    for transition in (
        store.tombstone(
            request.blob_id,
            expected_version=deleted.version,
            at=DELETED_AT + timedelta(microseconds=1),
        ),
        store.delete(
            request.blob_id,
            expected_version=deleted.version,
            at=DELETED_AT + timedelta(microseconds=1),
        ),
    ):
        with pytest.raises(BlobUnavailableError) as unavailable:
            await transition
        assert unavailable.value.state == "deleted"
    assert await store.metadata(request.blob_id) == deleted


async def test_partial_os_writes_are_retried_until_content_and_manifest_are_durable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "blobs"
    content = b"short writes must not truncate durable state"
    request = _request("blob-short-write", content)
    real_write = blob_owner.os.write

    def write_one_byte(fd: int, value: bytes | bytearray | memoryview) -> int:
        return real_write(fd, memoryview(value)[:1])

    monkeypatch.setattr(blob_owner.os, "write", write_one_byte)

    committed = await LocalBlobStore(root).put(request, content)

    assert committed.size_bytes == len(content)
    assert _object_path(root, content).read_bytes() == content
    assert await LocalBlobStore(root).metadata(request.blob_id) == committed


async def test_zero_progress_os_write_fails_without_publishing_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "blobs"
    content = b"a zero write cannot be treated as success"
    request = _request("blob-zero-write", content)
    real_write = blob_owner.os.write
    first_write = True

    def write_zero_once(fd: int, value: bytes | bytearray | memoryview) -> int:
        nonlocal first_write
        if first_write:
            first_write = False
            return 0
        return real_write(fd, value)

    monkeypatch.setattr(blob_owner.os, "write", write_zero_once)

    with pytest.raises(BlobStoreError, match="write|filesystem|progress"):
        await LocalBlobStore(root).put(request, content)

    assert await LocalBlobStore(root).metadata(request.blob_id) is None
    assert not _object_path(root, content).exists()
    assert not tuple(path for path in root.rglob("*") if path.name.endswith(".tmp"))


@pytest.mark.parametrize("managed_name", ("sha256", "records"))
async def test_symlinked_root_or_managed_directory_fails_closed(
    tmp_path: Path,
    managed_name: str,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "blobs"
    root.mkdir()
    (root / managed_name).symlink_to(outside, target_is_directory=True)
    content = b"managed directories must not escape the root"

    with pytest.raises(BlobStoreError, match="symlink"):
        await LocalBlobStore(root).put(_request("blob-managed-link", content), content)

    assert tuple(outside.iterdir()) == ()


async def test_symlinked_store_root_fails_closed_without_touching_target(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside-root"
    outside.mkdir()
    root = tmp_path / "blob-root-link"
    root.symlink_to(outside, target_is_directory=True)
    content = b"root links are not trusted"

    with pytest.raises(BlobStoreError, match="root.*symlink|symlink.*root"):
        await LocalBlobStore(root).put(_request("blob-root-link", content), content)

    assert tuple(outside.iterdir()) == ()


async def test_symlinked_manifest_fails_closed_without_reading_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"logical records cannot be symlinks"
    request = _request("blob-manifest-link", content)
    store = LocalBlobStore(root)
    await store.put(request, content)
    manifest = _only_private_manifest(root)
    outside = tmp_path / "outside-manifest.json"
    outside.write_text('{"untrusted":true}', encoding="utf-8")
    manifest.unlink()
    manifest.symlink_to(outside)

    with pytest.raises(BlobIntegrityError, match="symlink"):
        await LocalBlobStore(root).metadata(request.blob_id)

    assert outside.read_text(encoding="utf-8") == '{"untrusted":true}'


async def test_idempotent_retry_verifies_existing_content_before_success(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"retries must not bless corrupt objects"
    request = _request("blob-corrupt-retry", content)
    store = LocalBlobStore(root)
    committed = await store.put(request, content)
    _object_path(root, content).write_bytes(b"x" * len(content))

    with pytest.raises(BlobIntegrityError) as integrity:
        await LocalBlobStore(root).put(request, content)

    assert integrity.value.blob_id == request.blob_id
    assert integrity.value.digest == committed.digest
    assert await store.metadata(request.blob_id) == committed


@pytest.mark.parametrize(
    "damage",
    ("checksum", "extra", "missing", "duplicate", "identity-mismatch"),
)
async def test_manifest_decoder_rejects_tampering_before_returning_metadata(
    tmp_path: Path,
    damage: str,
) -> None:
    root = tmp_path / damage
    content = b"strict logical manifest"
    request = _request("blob-strict-manifest", content)
    store = LocalBlobStore(root)
    await store.put(request, content)
    manifest = _only_private_manifest(root)
    envelope = json.loads(manifest.read_text(encoding="utf-8"))
    assert isinstance(envelope, dict)
    record = envelope["record"]
    assert isinstance(record, dict)

    if damage == "checksum":
        envelope["checksum"] = "sha256:" + "0" * 64
        damaged = _canonical_json(envelope)
    elif damage == "extra":
        envelope["unexpected"] = True
        damaged = _canonical_json(envelope)
    elif damage == "missing":
        del record["retention_class"]
        damaged = _canonical_json(_checksummed_envelope(record))
    elif damage == "duplicate":
        checksum = envelope["checksum"]
        assert isinstance(checksum, str)
        damaged = (
            '{"checksum":'
            + _canonical_json(checksum)
            + ',"checksum":'
            + _canonical_json(checksum)
            + ',"record":'
            + _canonical_json(record)
            + "}"
        )
    else:
        record["blob_id"] = "blob-other-identity"
        damaged = _canonical_json(_checksummed_envelope(record))
    manifest.write_text(damaged, encoding="utf-8")

    with pytest.raises(BlobIntegrityError) as integrity:
        await LocalBlobStore(root).metadata(request.blob_id)

    assert integrity.value.blob_id == request.blob_id


async def test_reader_rejects_bool_size_and_serializes_concurrent_reads(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"aabbccdd"
    request = _request("blob-reader-adversarial", content)
    store = LocalBlobStore(root)
    await store.put(request, content)
    reader = await store.open(request.blob_id)

    with pytest.raises(TypeError, match="integer"):
        await reader.read(True)

    chunks = await asyncio.gather(*(reader.read(2) for _ in range(4)))
    assert sorted(chunks) == [b"aa", b"bb", b"cc", b"dd"]
    assert await reader.read(1) == b""
    await reader.close()
    await reader.close()
    with pytest.raises(ValueError, match="closed"):
        await reader.read(1)


async def test_changed_metadata_retry_is_rejected_without_object_publication(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"immutable record"
    changed = b"uncommitted contender"
    request = _request("blob-metadata-conflict", content)
    store = LocalBlobStore(root)
    committed = await store.put(request, content)

    with pytest.raises(BlobIdentityConflictError):
        await LocalBlobStore(root).put(
            replace(
                _request(request.blob_id, changed),
                media_type="text/plain",
            ),
            changed,
        )

    assert await store.metadata(request.blob_id) == committed
    assert not _object_path(root, changed).exists()


async def test_retry_restabilizes_content_after_rename_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "blobs"
    content = b"retry makes an uncertain content rename durable"
    request = _request("blob-content-restabilize", content)
    object_parent = _object_path(root, content).parent
    real_fsync_directory = blob_owner._fsync_directory
    failed = False

    def fail_content_directory_once(path: Path) -> None:
        nonlocal failed
        if path == object_parent and not failed:
            failed = True
            raise OSError("injected content directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        blob_owner,
        "_fsync_directory",
        fail_content_directory_once,
    )
    with pytest.raises(BlobStoreError, match="filesystem"):
        await LocalBlobStore(root).put(request, content)

    assert _object_path(root, content).read_bytes() == content
    assert await LocalBlobStore(root).metadata(request.blob_id) is None

    stabilized: list[Path] = []

    def record_fsync(path: Path) -> None:
        stabilized.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(blob_owner, "_fsync_directory", record_fsync)
    committed = await LocalBlobStore(root).put(request, content)

    assert committed.digest == _digest(content)
    assert object_parent in stabilized


async def test_retry_restabilizes_visible_manifest_after_directory_fsync_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "blobs"
    content = b"retry makes an uncertain manifest rename durable"
    request = _request("blob-manifest-restabilize", content)
    identity_hash = hashlib.sha256(request.blob_id.encode("utf-8")).hexdigest()
    manifest_parent = root / "records" / identity_hash[:2]
    real_fsync_directory = blob_owner._fsync_directory
    failed = False

    def fail_manifest_directory_once(path: Path) -> None:
        nonlocal failed
        if path == manifest_parent and not failed:
            failed = True
            raise OSError("injected manifest directory fsync failure")
        real_fsync_directory(path)

    monkeypatch.setattr(
        blob_owner,
        "_fsync_directory",
        fail_manifest_directory_once,
    )
    with pytest.raises(BlobStoreError, match="filesystem"):
        await LocalBlobStore(root).put(request, content)

    visible = await LocalBlobStore(root).metadata(request.blob_id)
    assert visible is not None

    stabilized: list[Path] = []

    def record_fsync(path: Path) -> None:
        stabilized.append(path)
        real_fsync_directory(path)

    monkeypatch.setattr(blob_owner, "_fsync_directory", record_fsync)
    retried = await LocalBlobStore(root).put(request, content)

    assert retried == visible
    assert manifest_parent in stabilized


async def test_deeply_nested_manifest_failure_stays_inside_integrity_boundary(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    content = b"recursive codec input"
    request = _request("blob-recursive-manifest", content)
    await LocalBlobStore(root).put(request, content)
    manifest = _only_private_manifest(root)
    manifest.write_bytes(b"[" * 2_000 + b"0" + b"]" * 2_000)

    with pytest.raises(BlobIntegrityError, match="manifest"):
        await LocalBlobStore(root).metadata(request.blob_id)


async def test_cancelled_reader_is_poisoned_after_definitive_hidden_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "blobs"
    content = b"cancelled reads cannot silently skip bytes"
    request = _request("blob-reader-cancelled", content)
    store = LocalBlobStore(root)
    await store.put(request, content)
    reader = await store.open(request.blob_id)
    real_await_sync_completion = blob_owner._await_sync_completion
    inject_cancellation = True

    async def cancelled_after_completion(callback: Any) -> tuple[Any, bool]:
        nonlocal inject_cancellation
        result, cancellation_requested = await real_await_sync_completion(callback)
        if inject_cancellation:
            inject_cancellation = False
            return result, True
        return result, cancellation_requested

    monkeypatch.setattr(
        blob_owner,
        "_await_sync_completion",
        cancelled_after_completion,
    )
    with pytest.raises(asyncio.CancelledError):
        await reader.read(8)
    with pytest.raises(ValueError, match="closed"):
        await reader.read(1)
