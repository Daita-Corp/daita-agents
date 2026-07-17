"""Portable content-addressed blob records and narrow storage contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import stat
import threading
from types import TracebackType
from typing import BinaryIO, Protocol, Self, TypeVar
from uuid import uuid4

from .._json import FrozenJsonObject, canonical_json

_SHA256_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_SHA256_PREFIX = re.compile(r"[0-9a-f]{2}\Z")
_MANIFEST_FORMAT_VERSION = 1
_MAX_MANIFEST_BYTES = 1_048_576
_DEFAULT_MAX_BLOB_BYTES = 16 * 1_024 * 1_024
_T = TypeVar("_T")

_ROOT_LOCKS_GUARD = threading.Lock()
_ROOT_LOCKS: dict[str, threading.RLock] = {}


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def _optional_identity(value: str | None, field_name: str) -> None:
    if value is not None:
        _required_text(value, field_name)


def _aware(value: datetime, field_name: str) -> None:
    if not isinstance(value, datetime):
        raise TypeError(f"{field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


def _digest(value: str, field_name: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if _SHA256_DIGEST.fullmatch(value) is None:
        raise ValueError(
            f"{field_name} must be sha256 followed by 64 lowercase hex characters"
        )


def _validate_provenance(
    operation_id: str | None,
    task_id: str | None,
    evidence_id: str | None,
) -> None:
    _optional_identity(operation_id, "blob operation_id")
    _optional_identity(task_id, "blob task_id")
    _optional_identity(evidence_id, "blob evidence_id")
    if task_id is not None and operation_id is None:
        raise ValueError("blob task provenance requires an operation_id")
    if evidence_id is not None and task_id is None:
        raise ValueError("blob evidence provenance requires a task_id")


class BlobStoreError(RuntimeError):
    """Base class for portable blob-store failures."""


class BlobNotFoundError(BlobStoreError):
    def __init__(self, blob_id: str) -> None:
        _required_text(blob_id, "blob_id")
        self.blob_id = blob_id
        super().__init__(f"unknown blob: {blob_id}")


class BlobIdentityConflictError(BlobStoreError):
    def __init__(self, blob_id: str) -> None:
        _required_text(blob_id, "blob_id")
        self.blob_id = blob_id
        super().__init__(
            f"blob identity is already committed with other facts: {blob_id}"
        )


class BlobRevisionConflict(BlobStoreError):
    def __init__(
        self,
        blob_id: str,
        *,
        expected_version: int,
        actual_version: int,
    ) -> None:
        _required_text(blob_id, "blob_id")
        self.blob_id = blob_id
        self.expected_version = expected_version
        self.actual_version = actual_version
        super().__init__(
            f"blob {blob_id} version conflict: expected {expected_version}, "
            f"found {actual_version}"
        )


class BlobUnavailableError(BlobStoreError):
    def __init__(self, blob_id: str, *, state: str) -> None:
        _required_text(blob_id, "blob_id")
        _required_text(state, "blob state")
        self.blob_id = blob_id
        self.state = state
        super().__init__(f"blob {blob_id} content is unavailable: {state}")


class BlobIntegrityError(BlobStoreError):
    def __init__(self, blob_id: str, *, digest: str | None, reason: str) -> None:
        _required_text(blob_id, "blob_id")
        if digest is not None:
            _digest(digest, "blob digest")
        _required_text(reason, "blob integrity reason")
        self.blob_id = blob_id
        self.digest = digest
        self.reason = reason
        super().__init__(f"blob {blob_id} failed integrity validation: {reason}")


class BlobRetentionError(BlobStoreError):
    def __init__(self, blob_id: str, *, reason: str) -> None:
        _required_text(blob_id, "blob_id")
        _required_text(reason, "blob retention reason")
        self.blob_id = blob_id
        self.reason = reason
        super().__init__(f"blob {blob_id} cannot be deleted: {reason}")


@dataclass(frozen=True, slots=True)
class BlobPut:
    """Immutable caller intent for one idempotent logical blob write."""

    blob_id: str
    media_type: str
    created_at: datetime
    sensitivity_class: str
    retention_class: str
    operation_id: str | None = None
    task_id: str | None = None
    evidence_id: str | None = None
    expected_digest: str | None = None
    encryption_metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _required_text(self.blob_id, "blob_id")
        _required_text(self.media_type, "blob media_type")
        _aware(self.created_at, "blob created_at")
        _required_text(self.sensitivity_class, "blob sensitivity_class")
        _required_text(self.retention_class, "blob retention_class")
        _validate_provenance(self.operation_id, self.task_id, self.evidence_id)
        if self.expected_digest is not None:
            _digest(self.expected_digest, "blob expected digest")
        object.__setattr__(
            self,
            "encryption_metadata",
            FrozenJsonObject.from_mapping(self.encryption_metadata),
        )


@dataclass(frozen=True, slots=True)
class BlobMetadata:
    """Durable logical metadata referencing one content-addressed object."""

    blob_id: str
    digest: str
    size_bytes: int
    media_type: str
    created_at: datetime
    sensitivity_class: str
    retention_class: str
    operation_id: str | None = None
    task_id: str | None = None
    evidence_id: str | None = None
    encryption_metadata: Mapping[str, object] = field(default_factory=dict)
    version: int = 1
    tombstoned_at: datetime | None = None
    deleted_at: datetime | None = None

    def __post_init__(self) -> None:
        _required_text(self.blob_id, "blob_id")
        _digest(self.digest, "blob digest")
        if not isinstance(self.size_bytes, int) or isinstance(self.size_bytes, bool):
            raise TypeError("blob size_bytes must be an integer")
        if self.size_bytes < 0:
            raise ValueError("blob size_bytes cannot be negative")
        _required_text(self.media_type, "blob media_type")
        _aware(self.created_at, "blob created_at")
        _required_text(self.sensitivity_class, "blob sensitivity_class")
        _required_text(self.retention_class, "blob retention_class")
        _validate_provenance(self.operation_id, self.task_id, self.evidence_id)
        if not isinstance(self.version, int) or isinstance(self.version, bool):
            raise TypeError("blob version must be an integer")
        if self.version < 1:
            raise ValueError("blob version must be positive")
        if self.tombstoned_at is not None:
            _aware(self.tombstoned_at, "blob tombstoned_at")
            if self.tombstoned_at < self.created_at:
                raise ValueError("blob tombstoned_at cannot precede created_at")
            if self.version < 2:
                raise ValueError("tombstoned blob metadata requires version 2 or later")
        if self.deleted_at is not None:
            _aware(self.deleted_at, "blob deleted_at")
            if self.tombstoned_at is None:
                raise ValueError("blob deletion requires a tombstone")
            if self.deleted_at < self.tombstoned_at:
                raise ValueError("blob deleted_at cannot precede its tombstone")
            if self.version < 3:
                raise ValueError("deleted blob metadata requires version 3 or later")
        object.__setattr__(
            self,
            "encryption_metadata",
            FrozenJsonObject.from_mapping(self.encryption_metadata),
        )


class BlobReader(Protocol):
    """Asynchronous reader that never exposes a filesystem path."""

    @property
    def metadata(self) -> BlobMetadata: ...

    async def read(self, size: int) -> bytes: ...

    async def close(self) -> None: ...

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...


class BlobStore(Protocol):
    """Persist content by hash while versioning logical metadata separately."""

    async def put(self, request: BlobPut, content: bytes) -> BlobMetadata: ...

    async def open(self, blob_id: str) -> BlobReader: ...

    async def metadata(self, blob_id: str) -> BlobMetadata | None: ...

    async def tombstone(
        self,
        blob_id: str,
        *,
        expected_version: int,
        at: datetime,
    ) -> BlobMetadata: ...

    async def delete(
        self,
        blob_id: str,
        *,
        expected_version: int,
        at: datetime,
    ) -> BlobMetadata: ...


class _LocalBlobReader:
    """One verified descriptor with serialized, cancellation-definitive reads."""

    def __init__(self, metadata: BlobMetadata, file: BinaryIO) -> None:
        self._metadata = metadata
        self._file = file
        self._lock = asyncio.Lock()
        self._closed = False

    @property
    def metadata(self) -> BlobMetadata:
        return self._metadata

    async def read(self, size: int) -> bytes:
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError("blob read size must be an integer")
        if size < 0:
            raise ValueError("blob read size must be non-negative")
        async with self._lock:
            if self._closed:
                raise ValueError("blob reader is closed")

            def read_bounded() -> bytes:
                remaining = max(0, self._metadata.size_bytes - self._file.tell())
                return self._file.read(min(size, remaining))

            result, cancellation_requested = await _await_sync_completion(read_bounded)
            if cancellation_requested:
                await _await_sync_completion(self._file.close)
                self._closed = True
        if cancellation_requested:
            raise asyncio.CancelledError
        return result

    async def close(self) -> None:
        acquisition_cancelled = await _acquire_lock_resistant(self._lock)
        close_cancelled = False
        try:
            if self._closed:
                if acquisition_cancelled:
                    raise asyncio.CancelledError
                return
            _, close_cancelled = await _await_sync_completion(self._file.close)
            self._closed = True
        finally:
            self._lock.release()
        if acquisition_cancelled or close_cancelled:
            raise asyncio.CancelledError

    async def __aenter__(self) -> Self:
        if self._closed:
            raise ValueError("blob reader is closed")
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    def _close_sync(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True


class LocalBlobStore:
    """Durable local content objects with separate logical manifests.

    Mutations are serialized across instances in this process. The embedded
    host's cross-process single-writer lease remains an Agent Home concern.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_blob_bytes: int = _DEFAULT_MAX_BLOB_BYTES,
    ) -> None:
        if not isinstance(root, (str, Path)):
            raise TypeError("local blob root must be a string or Path")
        if isinstance(root, str) and not root.strip():
            raise ValueError("local blob root must be non-empty")
        if not isinstance(max_blob_bytes, int) or isinstance(max_blob_bytes, bool):
            raise TypeError("max_blob_bytes must be an integer")
        if max_blob_bytes < 1:
            raise ValueError("max_blob_bytes must be positive")

        self._root = Path(os.path.abspath(os.fspath(root)))
        self._max_blob_bytes = max_blob_bytes
        self._mutation_lock = _root_lock(self._root)

    async def put(self, request: BlobPut, content: bytes) -> BlobMetadata:
        if not isinstance(request, BlobPut):
            raise TypeError("local blob request must be a BlobPut")
        if not isinstance(content, bytes):
            raise TypeError("local blob content must be bytes")
        result, cancellation_requested = await _await_sync_completion(
            lambda: self._locked(lambda: self._put_sync(request, content))
        )
        if cancellation_requested:
            raise asyncio.CancelledError
        return result

    async def open(self, blob_id: str) -> BlobReader:
        _required_text(blob_id, "blob_id")
        reader, cancellation_requested = await _await_sync_completion(
            lambda: self._locked(lambda: self._open_sync(blob_id))
        )
        if cancellation_requested:
            reader._close_sync()
            raise asyncio.CancelledError
        return reader

    async def metadata(self, blob_id: str) -> BlobMetadata | None:
        _required_text(blob_id, "blob_id")
        result, cancellation_requested = await _await_sync_completion(
            lambda: self._locked(lambda: self._load_record(blob_id))
        )
        if cancellation_requested:
            raise asyncio.CancelledError
        return result

    async def tombstone(
        self,
        blob_id: str,
        *,
        expected_version: int,
        at: datetime,
    ) -> BlobMetadata:
        _required_text(blob_id, "blob_id")
        _require_version(expected_version)
        _aware(at, "blob tombstone time")
        result, cancellation_requested = await _await_sync_completion(
            lambda: self._locked(
                lambda: self._tombstone_sync(
                    blob_id,
                    expected_version=expected_version,
                    at=at,
                )
            )
        )
        if cancellation_requested:
            raise asyncio.CancelledError
        return result

    async def delete(
        self,
        blob_id: str,
        *,
        expected_version: int,
        at: datetime,
    ) -> BlobMetadata:
        _required_text(blob_id, "blob_id")
        _require_version(expected_version)
        _aware(at, "blob deletion time")
        result, cancellation_requested = await _await_sync_completion(
            lambda: self._locked(
                lambda: self._delete_sync(
                    blob_id,
                    expected_version=expected_version,
                    at=at,
                )
            )
        )
        if cancellation_requested:
            raise asyncio.CancelledError
        return result

    async def cleanup_orphans(self, *, before: datetime) -> int:
        _aware(before, "blob orphan cutoff")
        result, cancellation_requested = await _await_sync_completion(
            lambda: self._locked(lambda: self._cleanup_orphans_sync(before))
        )
        if cancellation_requested:
            raise asyncio.CancelledError
        return result

    def _locked(self, callback: Callable[[], _T]) -> _T:
        with self._mutation_lock:
            try:
                self._ensure_layout()
                return callback()
            except (BlobStoreError, TypeError, ValueError):
                raise
            except OSError as error:
                raise BlobStoreError(
                    f"local blob filesystem operation failed under {self._root}: "
                    f"{error}"
                ) from error

    def _ensure_layout(self) -> None:
        _ensure_directory_tree(self._root, "local blob root")
        _ensure_directory(self._object_root, "local blob object directory")
        _ensure_directory(self._record_root, "local blob record directory")

    @property
    def _object_root(self) -> Path:
        return self._root / "sha256"

    @property
    def _record_root(self) -> Path:
        return self._root / "records"

    def _put_sync(self, request: BlobPut, content: bytes) -> BlobMetadata:
        if len(content) > self._max_blob_bytes:
            raise BlobStoreError(
                f"blob {request.blob_id} size {len(content)} exceeds maximum "
                f"{self._max_blob_bytes} bytes"
            )
        digest = "sha256:" + sha256(content).hexdigest()
        if request.expected_digest is not None and request.expected_digest != digest:
            raise BlobIntegrityError(
                request.blob_id,
                digest=digest,
                reason=(
                    "content digest does not match the caller's expected digest "
                    f"{request.expected_digest}"
                ),
            )

        existing = self._load_record(request.blob_id)
        if existing is not None:
            if not _request_matches_metadata(request, content, digest, existing):
                raise BlobIdentityConflictError(request.blob_id)
            if existing.tombstoned_at is not None or existing.deleted_at is not None:
                raise BlobIdentityConflictError(request.blob_id)
            self._stabilize_object(existing)
            self._stabilize_record(existing.blob_id)
            return existing

        metadata = BlobMetadata(
            blob_id=request.blob_id,
            digest=digest,
            size_bytes=len(content),
            media_type=request.media_type,
            created_at=request.created_at,
            sensitivity_class=request.sensitivity_class,
            retention_class=request.retention_class,
            operation_id=request.operation_id,
            task_id=request.task_id,
            evidence_id=request.evidence_id,
            encryption_metadata=request.encryption_metadata,
        )
        self._publish_content(metadata, content)
        self._write_record(metadata)
        return metadata

    def _open_sync(self, blob_id: str) -> _LocalBlobReader:
        metadata = self._load_record(blob_id)
        if metadata is None:
            raise BlobNotFoundError(blob_id)
        if metadata.deleted_at is not None:
            raise BlobUnavailableError(blob_id, state="deleted")
        if metadata.tombstoned_at is not None:
            raise BlobUnavailableError(blob_id, state="tombstoned")
        return _LocalBlobReader(metadata, self._open_verified_object(metadata))

    def _tombstone_sync(
        self,
        blob_id: str,
        *,
        expected_version: int,
        at: datetime,
    ) -> BlobMetadata:
        metadata = self._required_record(blob_id)
        _check_version(metadata, expected_version)
        if metadata.deleted_at is not None:
            raise BlobUnavailableError(blob_id, state="deleted")
        if metadata.tombstoned_at is not None:
            raise BlobUnavailableError(blob_id, state="tombstoned")
        updated = replace(
            metadata,
            version=metadata.version + 1,
            tombstoned_at=at,
        )
        self._write_record(updated)
        return updated

    def _delete_sync(
        self,
        blob_id: str,
        *,
        expected_version: int,
        at: datetime,
    ) -> BlobMetadata:
        metadata = self._required_record(blob_id)
        _check_version(metadata, expected_version)
        if metadata.deleted_at is not None:
            raise BlobUnavailableError(blob_id, state="deleted")
        if metadata.tombstoned_at is None:
            raise BlobRetentionError(blob_id, reason="blob must be tombstoned first")

        all_records = self._load_all_records()
        self._open_verified_object(metadata).close()
        shared = any(
            other.blob_id != blob_id
            and other.digest == metadata.digest
            and other.deleted_at is None
            for other in all_records
        )
        updated = replace(
            metadata,
            version=metadata.version + 1,
            deleted_at=at,
        )
        self._write_record(updated)
        if not shared:
            object_path = self._object_path(metadata.digest)
            os.unlink(object_path)
            _fsync_directory(object_path.parent)
        return updated

    def _cleanup_orphans_sync(self, before: datetime) -> int:
        records = self._load_all_records()
        referenced = {
            metadata.digest for metadata in records if metadata.deleted_at is None
        }
        cutoff = before.timestamp()
        object_candidates, object_temps = self._scan_object_tree()
        record_temps = self._scan_record_temps()
        external_temps = self._scan_temp_directories()

        stale_objects = [
            path
            for digest, path, modified_at in object_candidates
            if digest not in referenced and modified_at <= cutoff
        ]
        stale_temps = [
            path
            for path, modified_at in (*object_temps, *record_temps, *external_temps)
            if modified_at <= cutoff
        ]

        for path in stale_objects:
            _verify_orphan_object(path, maximum_bytes=self._max_blob_bytes)

        changed_directories: set[Path] = set()
        for path in (*stale_temps, *stale_objects):
            os.unlink(path)
            changed_directories.add(path.parent)
        for directory in sorted(changed_directories):
            _fsync_directory(directory)
        return len(stale_temps) + len(stale_objects)

    def _required_record(self, blob_id: str) -> BlobMetadata:
        metadata = self._load_record(blob_id)
        if metadata is None:
            raise BlobNotFoundError(blob_id)
        return metadata

    def _load_record(self, blob_id: str) -> BlobMetadata | None:
        path = self._record_path(blob_id)
        state = _path_state(path)
        if state == "missing":
            return None
        if state != "file":
            raise BlobIntegrityError(
                blob_id,
                digest=None,
                reason=f"logical manifest is a {state}, not a regular file",
            )
        raw = _read_regular_file(path, maximum_bytes=_MAX_MANIFEST_BYTES)
        metadata = _decode_manifest(raw, error_blob_id=blob_id)
        if metadata.blob_id != blob_id:
            raise BlobIntegrityError(
                blob_id,
                digest=None,
                reason="logical manifest blob_id does not match its lookup identity",
            )
        return metadata

    def _load_all_records(self) -> tuple[BlobMetadata, ...]:
        records: list[BlobMetadata] = []
        for path in self._record_files_and_temps()[0]:
            raw = _read_regular_file(path, maximum_bytes=_MAX_MANIFEST_BYTES)
            metadata = _decode_manifest(
                raw,
                error_blob_id=f"manifest-{path.stem}",
            )
            if path != self._record_path(metadata.blob_id):
                raise BlobIntegrityError(
                    metadata.blob_id,
                    digest=None,
                    reason="logical manifest is stored under the wrong identity hash",
                )
            records.append(metadata)
        return tuple(records)

    def _write_record(self, metadata: BlobMetadata) -> None:
        path = self._record_path(metadata.blob_id)
        _ensure_directory(path.parent, "local blob record prefix directory")
        content = _encode_manifest(metadata)
        temporary = path.parent / f".{path.stem}.{uuid4().hex}.tmp"
        try:
            _write_new_file(temporary, content)
            verified = _decode_manifest(
                _read_regular_file(temporary, maximum_bytes=_MAX_MANIFEST_BYTES),
                error_blob_id=metadata.blob_id,
            )
            if verified != metadata:
                raise BlobIntegrityError(
                    metadata.blob_id,
                    digest=metadata.digest,
                    reason="logical manifest verification changed canonical facts",
                )
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            _unlink_temporary(temporary)

    def _publish_content(self, metadata: BlobMetadata, content: bytes) -> None:
        path = self._object_path(metadata.digest)
        _ensure_directory(path.parent, "local blob object prefix directory")
        state = _path_state(path)
        if state != "missing":
            if state != "file":
                raise BlobIntegrityError(
                    metadata.blob_id,
                    digest=metadata.digest,
                    reason=f"content address is a {state}, not a regular file",
                )
            self._stabilize_object(metadata)
            return

        temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
        try:
            _write_new_file(temporary, content)
            _verify_file_digest(
                temporary,
                blob_id=metadata.blob_id,
                expected_digest=metadata.digest,
                expected_size=metadata.size_bytes,
            ).close()
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            _unlink_temporary(temporary)

    def _open_verified_object(self, metadata: BlobMetadata) -> BinaryIO:
        return _verify_file_digest(
            self._object_path(metadata.digest),
            blob_id=metadata.blob_id,
            expected_digest=metadata.digest,
            expected_size=metadata.size_bytes,
        )

    def _stabilize_object(self, metadata: BlobMetadata) -> None:
        file = self._open_verified_object(metadata)
        try:
            os.fsync(file.fileno())
        finally:
            file.close()
        _fsync_directory(self._object_path(metadata.digest).parent)

    def _stabilize_record(self, blob_id: str) -> None:
        path = self._record_path(blob_id)
        descriptor = _open_regular_descriptor(path)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _fsync_directory(path.parent)

    def _record_path(self, blob_id: str) -> Path:
        identity_hash = sha256(blob_id.encode("utf-8")).hexdigest()
        return self._record_root / identity_hash[:2] / f"{identity_hash}.json"

    def _object_path(self, digest: str) -> Path:
        digest_hex = digest.removeprefix("sha256:")
        return self._object_root / digest_hex[:2] / digest_hex

    def _record_files_and_temps(
        self,
    ) -> tuple[tuple[Path, ...], tuple[tuple[Path, float], ...]]:
        files: list[Path] = []
        temps: list[tuple[Path, float]] = []
        for current, directory_names, file_names in os.walk(
            self._record_root,
            followlinks=False,
        ):
            current_path = Path(current)
            for name in directory_names:
                candidate = current_path / name
                state = _path_state(candidate)
                if state != "directory":
                    raise BlobStoreError(
                        f"local blob record tree contains {state}: {candidate}"
                    )
            for name in file_names:
                candidate = current_path / name
                state = _path_state(candidate)
                if state != "file":
                    raise BlobStoreError(
                        f"local blob record tree contains {state}: {candidate}"
                    )
                modified_at = os.lstat(candidate).st_mtime
                if _is_temporary_name(name):
                    temps.append((candidate, modified_at))
                elif name.endswith(".json"):
                    files.append(candidate)
                else:
                    raise BlobStoreError(
                        f"local blob record tree contains an unknown file: {candidate}"
                    )
        return tuple(sorted(files)), tuple(sorted(temps))

    def _scan_record_temps(self) -> tuple[tuple[Path, float], ...]:
        return self._record_files_and_temps()[1]

    def _scan_object_tree(
        self,
    ) -> tuple[
        tuple[tuple[str, Path, float], ...],
        tuple[tuple[Path, float], ...],
    ]:
        objects: list[tuple[str, Path, float]] = []
        temps: list[tuple[Path, float]] = []
        for prefix_entry in os.scandir(self._object_root):
            prefix_path = Path(prefix_entry.path)
            state = _path_state(prefix_path)
            if state != "directory":
                raise BlobStoreError(
                    f"local blob object tree contains {state}: {prefix_path}"
                )
            if _SHA256_PREFIX.fullmatch(prefix_entry.name) is None:
                raise BlobStoreError(
                    f"local blob object tree contains an unknown prefix: {prefix_path}"
                )
            for object_entry in os.scandir(prefix_path):
                path = Path(object_entry.path)
                object_state = _path_state(path)
                if object_state != "file":
                    raise BlobStoreError(
                        f"local blob object tree contains {object_state}: {path}"
                    )
                modified_at = os.lstat(path).st_mtime
                if _is_temporary_name(object_entry.name):
                    temps.append((path, modified_at))
                    continue
                if (
                    _SHA256_HEX.fullmatch(object_entry.name) is None
                    or object_entry.name[:2] != prefix_entry.name
                ):
                    raise BlobStoreError(
                        f"local blob object tree contains an unknown file: {path}"
                    )
                objects.append(("sha256:" + object_entry.name, path, modified_at))
        return tuple(sorted(objects)), tuple(sorted(temps))

    def _scan_temp_directories(self) -> tuple[tuple[Path, float], ...]:
        candidates: list[tuple[Path, float]] = []
        for directory in (self._root / "tmp", self._root / ".tmp"):
            state = _path_state(directory)
            if state == "missing":
                continue
            if state != "directory":
                raise BlobStoreError(
                    f"local blob temporary path is a {state}: {directory}"
                )
            for current, directory_names, file_names in os.walk(
                directory,
                followlinks=False,
            ):
                current_path = Path(current)
                for name in directory_names:
                    path = current_path / name
                    child_state = _path_state(path)
                    if child_state != "directory":
                        raise BlobStoreError(
                            f"local blob temporary tree contains {child_state}: {path}"
                        )
                for name in file_names:
                    path = current_path / name
                    child_state = _path_state(path)
                    if child_state != "file":
                        raise BlobStoreError(
                            f"local blob temporary tree contains {child_state}: {path}"
                        )
                    candidates.append((path, os.lstat(path).st_mtime))
        return tuple(sorted(candidates))


def _root_lock(root: Path) -> threading.RLock:
    key = os.path.normcase(os.fspath(root))
    with _ROOT_LOCKS_GUARD:
        lock = _ROOT_LOCKS.get(key)
        if lock is None:
            lock = threading.RLock()
            _ROOT_LOCKS[key] = lock
        return lock


async def _await_sync_completion(
    callback: Callable[[], _T],
) -> tuple[_T, bool]:
    """Finish one offloaded filesystem transaction before honoring cancellation."""

    worker = asyncio.create_task(asyncio.to_thread(callback))
    cancellation_requested = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancellation_requested = True
            continue
        except BaseException:
            break
    try:
        result = worker.result()
    except BaseException as error:
        if cancellation_requested and not isinstance(error, asyncio.CancelledError):
            raise asyncio.CancelledError from error
        raise
    return result, cancellation_requested


async def _acquire_lock_resistant(lock: asyncio.Lock) -> bool:
    """Acquire a cleanup lock before propagating caller cancellation."""

    acquisition = asyncio.create_task(lock.acquire())
    cancellation_requested = False
    while not acquisition.done():
        try:
            await asyncio.shield(acquisition)
        except asyncio.CancelledError:
            cancellation_requested = True
            continue
        except BaseException:
            break
    try:
        acquisition.result()
    except BaseException as error:
        if cancellation_requested and not isinstance(error, asyncio.CancelledError):
            raise asyncio.CancelledError from error
        raise
    return cancellation_requested


def _require_version(value: int) -> None:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError("expected blob version must be an integer")
    if value < 1:
        raise ValueError("expected blob version must be positive")


def _check_version(metadata: BlobMetadata, expected_version: int) -> None:
    if metadata.version != expected_version:
        raise BlobRevisionConflict(
            metadata.blob_id,
            expected_version=expected_version,
            actual_version=metadata.version,
        )


def _request_matches_metadata(
    request: BlobPut,
    content: bytes,
    digest: str,
    metadata: BlobMetadata,
) -> bool:
    return (
        metadata.blob_id == request.blob_id
        and metadata.digest == digest
        and metadata.size_bytes == len(content)
        and metadata.media_type == request.media_type
        and metadata.created_at == request.created_at
        and metadata.sensitivity_class == request.sensitivity_class
        and metadata.retention_class == request.retention_class
        and metadata.operation_id == request.operation_id
        and metadata.task_id == request.task_id
        and metadata.evidence_id == request.evidence_id
        and metadata.encryption_metadata == request.encryption_metadata
    )


def _require_directory(path: Path, label: str) -> None:
    state = _path_state(path)
    if state != "directory":
        raise BlobStoreError(f"{label} is a {state}: {path}")


def _ensure_directory_tree(path: Path, label: str) -> None:
    missing: list[Path] = []
    candidate = path
    while _path_state(candidate) == "missing":
        missing.append(candidate)
        parent = candidate.parent
        if parent == candidate:
            break
        candidate = parent
    _require_directory(candidate, f"{label} ancestor")
    for directory in reversed(missing):
        os.mkdir(directory, mode=0o700)
        _fsync_directory(directory.parent)
    _require_directory(path, label)


def _ensure_directory(path: Path, label: str) -> None:
    try:
        os.mkdir(path, mode=0o700)
    except FileExistsError:
        pass
    else:
        _fsync_directory(path.parent)
    _require_directory(path, label)


def _path_state(path: Path) -> str:
    try:
        mode = os.lstat(path).st_mode
    except FileNotFoundError:
        return "missing"
    if stat.S_ISLNK(mode):
        return "symlink"
    if stat.S_ISREG(mode):
        return "file"
    if stat.S_ISDIR(mode):
        return "directory"
    return "unsupported filesystem object"


def _write_new_file(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(content)
        offset = 0
        while offset < len(view):
            written = os.write(descriptor, view[offset:])
            if written <= 0:
                raise OSError("blob write made no progress")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _read_regular_file(path: Path, *, maximum_bytes: int) -> bytes:
    descriptor = _open_regular_descriptor(path)
    try:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, maximum_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > maximum_bytes:
                raise BlobStoreError(f"local blob record is too large: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _open_regular_descriptor(path: Path) -> int:
    state = _path_state(path)
    if state != "file":
        raise BlobStoreError(f"expected a regular file, found {state}: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    status = os.fstat(descriptor)
    if not stat.S_ISREG(status.st_mode):
        os.close(descriptor)
        raise BlobStoreError(f"opened blob path is not a regular file: {path}")
    if status.st_nlink != 1:
        os.close(descriptor)
        raise BlobStoreError(f"opened blob path has an unsafe hard-link count: {path}")
    return descriptor


def _verify_file_digest(
    path: Path,
    *,
    blob_id: str,
    expected_digest: str,
    expected_size: int,
) -> BinaryIO:
    try:
        descriptor = _open_regular_descriptor(path)
    except (BlobStoreError, OSError) as error:
        raise BlobIntegrityError(
            blob_id,
            digest=expected_digest,
            reason=f"content object is unavailable or unsafe: {error}",
        ) from error
    file = os.fdopen(descriptor, "rb", closefd=True)
    try:
        actual_size = os.fstat(file.fileno()).st_size
        if actual_size != expected_size:
            raise BlobIntegrityError(
                blob_id,
                digest=expected_digest,
                reason=(
                    f"content object has size {actual_size}; expected "
                    f"{expected_size}"
                ),
            )
        digest = sha256()
        size = 0
        while True:
            chunk = file.read(65_536)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
        actual_digest = "sha256:" + digest.hexdigest()
        if size != expected_size or actual_digest != expected_digest:
            raise BlobIntegrityError(
                blob_id,
                digest=expected_digest,
                reason=(
                    f"content object has size {size} and digest {actual_digest}; "
                    f"expected size {expected_size} and digest {expected_digest}"
                ),
            )
        file.seek(0)
        return file
    except BaseException:
        file.close()
        raise


def _verify_orphan_object(path: Path, *, maximum_bytes: int) -> None:
    expected_hex = path.name
    size = os.lstat(path).st_size
    if size > maximum_bytes:
        raise BlobIntegrityError(
            f"orphan-{expected_hex}",
            digest="sha256:" + expected_hex,
            reason=(
                f"orphan content size {size} exceeds maintenance limit "
                f"{maximum_bytes}"
            ),
        )
    file = _verify_file_digest(
        path,
        blob_id=f"orphan-{expected_hex}",
        expected_digest="sha256:" + expected_hex,
        expected_size=size,
    )
    file.close()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_temporary(path: Path) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _is_temporary_name(name: str) -> bool:
    return name.startswith(".") and name.endswith(".tmp")


_MANIFEST_FIELDS = {
    "blob_id",
    "created_at",
    "deleted_at",
    "digest",
    "encryption_metadata",
    "evidence_id",
    "format_version",
    "media_type",
    "operation_id",
    "retention_class",
    "sensitivity_class",
    "size_bytes",
    "task_id",
    "tombstoned_at",
    "version",
}


def _encode_manifest(metadata: BlobMetadata) -> bytes:
    record = {
        "blob_id": metadata.blob_id,
        "created_at": _encode_datetime(metadata.created_at),
        "deleted_at": _encode_optional_datetime(metadata.deleted_at),
        "digest": metadata.digest,
        "encryption_metadata": metadata.encryption_metadata,
        "evidence_id": metadata.evidence_id,
        "format_version": _MANIFEST_FORMAT_VERSION,
        "media_type": metadata.media_type,
        "operation_id": metadata.operation_id,
        "retention_class": metadata.retention_class,
        "sensitivity_class": metadata.sensitivity_class,
        "size_bytes": metadata.size_bytes,
        "task_id": metadata.task_id,
        "tombstoned_at": _encode_optional_datetime(metadata.tombstoned_at),
        "version": metadata.version,
    }
    record_json = canonical_json(record)
    envelope = {
        "checksum": "sha256:" + sha256(record_json.encode("utf-8")).hexdigest(),
        "record": record,
    }
    return canonical_json(envelope).encode("utf-8")


def _decode_manifest(raw: bytes, *, error_blob_id: str) -> BlobMetadata:
    try:
        text = raw.decode("utf-8")
        decoded = json.loads(
            text,
            parse_constant=_reject_json_constant,
            object_pairs_hook=_reject_duplicate_json_keys,
        )
        if canonical_json(decoded) != text:
            raise ValueError("manifest JSON is not canonical")
        if not isinstance(decoded, dict) or set(decoded) != {"checksum", "record"}:
            raise ValueError("manifest envelope has unknown or missing fields")
        checksum = decoded["checksum"]
        record = decoded["record"]
        if not isinstance(checksum, str) or _SHA256_DIGEST.fullmatch(checksum) is None:
            raise ValueError("manifest checksum is not canonical sha256")
        if not isinstance(record, dict) or set(record) != _MANIFEST_FIELDS:
            raise ValueError("manifest record has unknown or missing fields")
        record_json = canonical_json(record)
        actual_checksum = "sha256:" + sha256(record_json.encode("utf-8")).hexdigest()
        if checksum != actual_checksum:
            raise ValueError("manifest checksum does not match its record")
        if (
            _manifest_integer(record["format_version"], "format_version")
            != _MANIFEST_FORMAT_VERSION
        ):
            raise ValueError("unsupported local blob manifest format version")
        encryption_metadata = record["encryption_metadata"]
        if not isinstance(encryption_metadata, dict):
            raise TypeError("manifest encryption_metadata must be an object")
        return BlobMetadata(
            blob_id=_manifest_text(record["blob_id"], "blob_id"),
            digest=_manifest_text(record["digest"], "digest"),
            size_bytes=_manifest_integer(record["size_bytes"], "size_bytes"),
            media_type=_manifest_text(record["media_type"], "media_type"),
            created_at=_decode_datetime(record["created_at"], "created_at"),
            sensitivity_class=_manifest_text(
                record["sensitivity_class"],
                "sensitivity_class",
            ),
            retention_class=_manifest_text(
                record["retention_class"],
                "retention_class",
            ),
            operation_id=_manifest_optional_text(record["operation_id"]),
            task_id=_manifest_optional_text(record["task_id"]),
            evidence_id=_manifest_optional_text(record["evidence_id"]),
            encryption_metadata=encryption_metadata,
            version=_manifest_integer(record["version"], "version"),
            tombstoned_at=_decode_optional_datetime(
                record["tombstoned_at"],
                "tombstoned_at",
            ),
            deleted_at=_decode_optional_datetime(record["deleted_at"], "deleted_at"),
        )
    except BlobIntegrityError:
        raise
    except (
        TypeError,
        ValueError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
        OverflowError,
    ) as error:
        raise BlobIntegrityError(
            error_blob_id,
            digest=None,
            reason=f"logical manifest is invalid: {error}",
        ) from error


def _manifest_text(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"manifest {label} must be text")
    return value


def _manifest_optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("manifest optional identity must be text or null")
    return value


def _manifest_integer(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"manifest {label} must be an integer")
    return value


def _encode_datetime(value: datetime) -> str:
    return (
        value.astimezone(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _encode_optional_datetime(value: datetime | None) -> str | None:
    return None if value is None else _encode_datetime(value)


def _decode_datetime(value: object, label: str) -> datetime:
    if not isinstance(value, str):
        raise TypeError(f"manifest {label} must be text")
    decoded = datetime.fromisoformat(value.replace("Z", "+00:00"))
    _aware(decoded, f"manifest {label}")
    if _encode_datetime(decoded) != value:
        raise ValueError(f"manifest {label} is not in canonical UTC form")
    return decoded


def _decode_optional_datetime(value: object, label: str) -> datetime | None:
    if value is None:
        return None
    return _decode_datetime(value, label)


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result
