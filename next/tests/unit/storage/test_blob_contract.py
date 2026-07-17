from __future__ import annotations

import inspect
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from daita._json import FrozenJsonObject
from daita.storage import blobs as blob_owner
from daita.storage.blobs import (
    BlobIdentityConflictError,
    BlobIntegrityError,
    BlobMetadata,
    BlobNotFoundError,
    BlobPut,
    BlobReader,
    BlobRetentionError,
    BlobRevisionConflict,
    BlobStore,
    BlobStoreError,
    BlobUnavailableError,
)

NOW = datetime(2026, 7, 17, 15, 0, tzinfo=timezone.utc)
DIGEST = f"sha256:{'a' * 64}"


def _put() -> BlobPut:
    return BlobPut(
        blob_id="blob-1",
        media_type="application/octet-stream",
        created_at=NOW,
        sensitivity_class="internal",
        retention_class="operation",
        operation_id="operation-1",
        task_id="task-1",
        evidence_id="evidence-1",
        expected_digest=DIGEST,
        encryption_metadata={"cipher": {"key_ids": ["key-1"]}},
    )


def _metadata() -> BlobMetadata:
    return BlobMetadata(
        blob_id="blob-1",
        media_type="application/octet-stream",
        created_at=NOW,
        sensitivity_class="internal",
        retention_class="operation",
        operation_id="operation-1",
        task_id="task-1",
        evidence_id="evidence-1",
        encryption_metadata={"cipher": {"key_ids": ["key-1"]}},
        digest=DIGEST,
        size_bytes=0,
        version=1,
        tombstoned_at=None,
        deleted_at=None,
    )


def _replace_record(
    record: BlobPut | BlobMetadata,
    **changes: Any,
) -> BlobPut | BlobMetadata:
    return replace(record, **changes)


@pytest.mark.parametrize(
    "field_name",
    (
        "blob_id",
        "media_type",
        "sensitivity_class",
        "retention_class",
        "operation_id",
        "task_id",
        "evidence_id",
    ),
)
def test_blob_records_reject_blank_identity_and_classification_labels(
    field_name: str,
) -> None:
    for record in (_put(), _metadata()):
        with pytest.raises(ValueError, match="non-empty"):
            _replace_record(record, **{field_name: " "})


def test_blob_records_require_nested_provenance() -> None:
    for record in (_put(), _metadata()):
        with pytest.raises(ValueError, match="task.*operation|operation.*task"):
            replace(record, operation_id=None)
        with pytest.raises(ValueError, match="evidence.*task|task.*evidence"):
            replace(record, task_id=None)

        assert replace(record, task_id=None, evidence_id=None).operation_id is not None
        detached = replace(
            record,
            operation_id=None,
            task_id=None,
            evidence_id=None,
        )
        assert detached.operation_id is detached.task_id is detached.evidence_id is None


def test_blob_records_require_aware_ordered_lifecycle_times() -> None:
    naive = NOW.replace(tzinfo=None)
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(_put(), created_at=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(_metadata(), tombstoned_at=naive)
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(_metadata(), deleted_at=naive)

    tombstoned = replace(
        _metadata(),
        version=2,
        tombstoned_at=NOW + timedelta(seconds=1),
    )
    deleted = replace(
        tombstoned,
        version=3,
        deleted_at=NOW + timedelta(seconds=2),
    )
    assert deleted.deleted_at is not None

    with pytest.raises(ValueError, match="tombstone|created_at"):
        replace(_metadata(), tombstoned_at=NOW - timedelta(microseconds=1))
    with pytest.raises(ValueError, match="delet.*tombstone"):
        replace(_metadata(), deleted_at=NOW + timedelta(seconds=1))
    with pytest.raises(ValueError, match="deleted_at|tombstone"):
        replace(
            _metadata(),
            version=3,
            tombstoned_at=NOW + timedelta(seconds=2),
            deleted_at=NOW + timedelta(seconds=1),
        )


def test_blob_metadata_versions_cover_committed_lifecycle_transitions() -> None:
    with pytest.raises(ValueError, match="version 2"):
        replace(_metadata(), tombstoned_at=NOW + timedelta(seconds=1))

    with pytest.raises(ValueError, match="version 3"):
        replace(
            _metadata(),
            version=2,
            tombstoned_at=NOW + timedelta(seconds=1),
            deleted_at=NOW + timedelta(seconds=2),
        )


@pytest.mark.parametrize(
    "digest",
    (
        "a" * 64,
        f"sha256:{'A' * 64}",
        f"sha256:{'a' * 63}",
        f"sha512:{'a' * 64}",
        f" sha256:{'a' * 64}",
    ),
)
def test_blob_digests_are_exact_lowercase_sha256_values(digest: str) -> None:
    with pytest.raises(ValueError, match="digest|sha256"):
        replace(_put(), expected_digest=digest)
    with pytest.raises(ValueError, match="digest|sha256"):
        replace(_metadata(), digest=digest)

    assert replace(_put(), expected_digest=None).expected_digest is None


@pytest.mark.parametrize(
    ("field_name", "value"),
    (("size_bytes", -1), ("size_bytes", True), ("version", 0), ("version", True)),
)
def test_blob_metadata_rejects_invalid_integer_fields(
    field_name: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field_name):
        _replace_record(_metadata(), **{field_name: value})


def test_encryption_metadata_is_recursively_immutable_and_detached() -> None:
    key_ids = ["key-1"]
    source: dict[str, object] = {"cipher": {"key_ids": key_ids}}
    request = replace(_put(), encryption_metadata=source)
    metadata = replace(_metadata(), encryption_metadata=source)
    key_ids.append("key-2")

    for record in (request, metadata):
        assert isinstance(record.encryption_metadata, FrozenJsonObject)
        cipher = record.encryption_metadata["cipher"]
        assert isinstance(cipher, FrozenJsonObject)
        assert cipher["key_ids"] == ("key-1",)

    with pytest.raises(TypeError, match="Unsupported JSON value"):
        replace(_put(), encryption_metadata={"unsafe": object()})


def test_blob_protocols_have_only_the_narrow_async_surface() -> None:
    assert isinstance(inspect.getattr_static(BlobReader, "metadata"), property)
    assert {name for name in BlobReader.__dict__ if not name.startswith("_")} == {
        "metadata",
        "read",
        "close",
    }
    assert {name for name in BlobStore.__dict__ if not name.startswith("_")} == {
        "put",
        "open",
        "metadata",
        "tombstone",
        "delete",
    }

    methods = (
        BlobReader.read,
        BlobReader.close,
        BlobReader.__aenter__,
        BlobReader.__aexit__,
        BlobStore.put,
        BlobStore.open,
        BlobStore.metadata,
        BlobStore.tombstone,
        BlobStore.delete,
    )
    assert all(inspect.iscoroutinefunction(method) for method in methods)
    assert tuple(inspect.signature(BlobReader.read).parameters) == ("self", "size")
    assert (
        inspect.signature(BlobReader.read).parameters["size"].default
        is inspect.Parameter.empty
    )
    assert tuple(inspect.signature(BlobReader.close).parameters) == ("self",)
    assert tuple(inspect.signature(BlobStore.put).parameters) == (
        "self",
        "request",
        "content",
    )
    assert tuple(inspect.signature(BlobStore.open).parameters) == ("self", "blob_id")
    assert tuple(inspect.signature(BlobStore.metadata).parameters) == (
        "self",
        "blob_id",
    )
    assert tuple(inspect.signature(BlobStore.tombstone).parameters) == (
        "self",
        "blob_id",
        "expected_version",
        "at",
    )
    assert tuple(inspect.signature(BlobStore.delete).parameters) == (
        "self",
        "blob_id",
        "expected_version",
        "at",
    )


def test_blob_module_does_not_grow_a_generic_state_store() -> None:
    assert not hasattr(blob_owner, "StateStore")


def test_blob_store_errors_are_typed_and_preserve_machine_readable_facts() -> None:
    missing = BlobNotFoundError("blob-1")
    identity = BlobIdentityConflictError("blob-1")
    revision = BlobRevisionConflict(
        "blob-1",
        expected_version=2,
        actual_version=3,
    )
    unavailable = BlobUnavailableError("blob-1", state="tombstoned")
    integrity = BlobIntegrityError(
        "blob-1",
        digest=DIGEST,
        reason="content hash mismatch",
    )
    corrupt_record = BlobIntegrityError(
        "blob-1",
        digest=None,
        reason="metadata is not canonical",
    )
    retention = BlobRetentionError("blob-1", reason="content is still referenced")

    assert all(
        isinstance(error, BlobStoreError)
        for error in (
            missing,
            identity,
            revision,
            unavailable,
            integrity,
            corrupt_record,
            retention,
        )
    )
    assert missing.blob_id == identity.blob_id == "blob-1"
    assert revision.blob_id == "blob-1"
    assert revision.expected_version == 2
    assert revision.actual_version == 3
    assert unavailable.state == "tombstoned"
    assert integrity.digest == DIGEST
    assert integrity.reason == "content hash mismatch"
    assert corrupt_record.digest is None
    assert retention.reason == "content is still referenced"
