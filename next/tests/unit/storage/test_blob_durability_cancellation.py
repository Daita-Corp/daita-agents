from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import os
from pathlib import Path
import threading
from typing import Callable

import pytest

from daita.storage import blobs as blob_owner
from daita.storage.blobs import (
    BlobIntegrityError,
    BlobPut,
    BlobStoreError,
    LocalBlobStore,
)

CONTENT = b"durable local blob content\x00\xff"
NOW = datetime(2026, 7, 18, 12, 30, 45, 123_456, tzinfo=timezone.utc)


def _digest(content: bytes) -> str:
    return "sha256:" + hashlib.sha256(content).hexdigest()


def _content_path(root: Path, content: bytes = CONTENT) -> Path:
    digest_hex = hashlib.sha256(content).hexdigest()
    return root / "sha256" / digest_hex[:2] / digest_hex


def _put(blob_id: str, *, expected_digest: str | None = None) -> BlobPut:
    return BlobPut(
        blob_id=blob_id,
        media_type="application/octet-stream",
        created_at=NOW,
        sensitivity_class="internal",
        retention_class="operation",
        operation_id="operation-durability",
        task_id="task-durability",
        evidence_id="evidence-durability",
        expected_digest=expected_digest,
        encryption_metadata={"scheme": None},
    )


def _temp_files(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    found: set[Path] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_file() and (
            "tmp" in {part.lower().lstrip(".") for part in relative.parts}
            or ".tmp" in path.name.lower()
        ):
            found.add(relative)
    return found


def _tree_snapshot(root: Path) -> tuple[tuple[str, str, int, int, bytes], ...]:
    if not root.exists():
        return ()
    snapshot: list[tuple[str, str, int, int, bytes]] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        stat = path.lstat()
        if path.is_symlink():
            snapshot.append(
                (
                    relative,
                    "symlink",
                    stat.st_mode,
                    stat.st_mtime_ns,
                    os.readlink(path).encode(),
                )
            )
        elif path.is_dir():
            snapshot.append(
                (relative, "directory", stat.st_mode, stat.st_mtime_ns, b"")
            )
        else:
            snapshot.append(
                (relative, "file", stat.st_mode, stat.st_mtime_ns, path.read_bytes())
            )
    return tuple(snapshot)


async def _assert_cancelled_worker_finishes(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    install_blocker: Callable[[threading.Event, threading.Event], None],
    *,
    blob_id: str,
) -> None:
    entered = threading.Event()
    release = threading.Event()
    install_blocker(entered, release)
    store = LocalBlobStore(root, max_blob_bytes=1024)
    task = asyncio.create_task(store.put(_put(blob_id), CONTENT))

    assert await asyncio.to_thread(entered.wait, 5), "worker did not reach fault seam"
    task.cancel()
    await asyncio.sleep(0)
    cancellation_escaped_while_worker_was_blocked = task.done()
    release.set()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)

    assert cancellation_escaped_while_worker_was_blocked is False
    assert _temp_files(root) == set()
    assert _content_path(root).read_bytes() == CONTENT
    reopened = LocalBlobStore(root, max_blob_bytes=1024)
    metadata = await reopened.metadata(blob_id)
    assert metadata is not None
    assert metadata.digest == _digest(CONTENT)

    completed = _tree_snapshot(root)
    await asyncio.sleep(0.05)
    assert _tree_snapshot(root) == completed


async def test_cancellation_during_worker_write_waits_for_definitive_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_write = blob_owner.os.write

    def install(entered: threading.Event, release: threading.Event) -> None:
        blocked = False

        def blocking_write(fd: int, content: bytes | bytearray | memoryview) -> int:
            nonlocal blocked
            if not blocked:
                blocked = True
                entered.set()
                assert release.wait(5), "test did not release the worker write"
            return real_write(fd, content)

        monkeypatch.setattr(blob_owner.os, "write", blocking_write)

    await _assert_cancelled_worker_finishes(
        tmp_path / "blobs",
        monkeypatch,
        install,
        blob_id="blob-cancel-write",
    )


async def test_cancellation_during_content_rename_waits_for_definitive_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "blobs"
    content_path = _content_path(root)
    real_replace = blob_owner.os.replace

    def install(entered: threading.Event, release: threading.Event) -> None:
        blocked = False

        def blocking_replace(
            source: os.PathLike[str] | str, target: os.PathLike[str] | str
        ) -> None:
            nonlocal blocked
            if Path(target) == content_path and not blocked:
                blocked = True
                entered.set()
                assert release.wait(5), "test did not release the content rename"
            real_replace(source, target)

        monkeypatch.setattr(blob_owner.os, "replace", blocking_replace)

    await _assert_cancelled_worker_finishes(
        root,
        monkeypatch,
        install,
        blob_id="blob-cancel-rename",
    )


@pytest.mark.parametrize(
    "fault",
    ("write", "fsync", "hash", "content-rename", "manifest-publication"),
)
async def test_failed_put_never_publishes_metadata_or_corrupt_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fault: str,
) -> None:
    root = tmp_path / "blobs"
    target = _content_path(root)
    request = _put("blob-failed-" + fault, expected_digest=_digest(CONTENT))

    if fault == "write":
        real_write = blob_owner.os.write
        failed = False

        def fail_write(fd: int, content: bytes | bytearray | memoryview) -> int:
            nonlocal failed
            if not failed:
                failed = True
                if content:
                    real_write(fd, memoryview(content)[:1])
                raise OSError("injected blob write failure")
            return real_write(fd, content)

        monkeypatch.setattr(blob_owner.os, "write", fail_write)
    elif fault == "fsync":
        real_fsync = blob_owner.os.fsync
        failed = False

        def fail_fsync(fd: int) -> None:
            nonlocal failed
            if not failed:
                failed = True
                raise OSError("injected blob fsync failure")
            real_fsync(fd)

        monkeypatch.setattr(blob_owner.os, "fsync", fail_fsync)
    elif fault == "hash":
        request = _put("blob-failed-hash", expected_digest="sha256:" + "0" * 64)
    else:
        real_replace = blob_owner.os.replace

        def fail_replace(
            source: os.PathLike[str] | str, destination: os.PathLike[str] | str
        ) -> None:
            is_content_publication = Path(destination) == target
            should_fail = (fault == "content-rename" and is_content_publication) or (
                fault == "manifest-publication" and not is_content_publication
            )
            if should_fail:
                raise OSError(f"injected {fault} failure")
            real_replace(source, destination)

        monkeypatch.setattr(blob_owner.os, "replace", fail_replace)

    store = LocalBlobStore(root, max_blob_bytes=1024)
    error_type = BlobIntegrityError if fault == "hash" else BlobStoreError
    with pytest.raises(error_type):
        await store.put(request, CONTENT)

    assert await store.metadata(request.blob_id) is None
    assert (
        await LocalBlobStore(root, max_blob_bytes=1024).metadata(request.blob_id)
        is None
    )
    assert _temp_files(root) == set()
    if target.exists():
        assert target.is_file() and not target.is_symlink()
        assert target.read_bytes() == CONTENT


def _write_orphan(root: Path, content: bytes, *, modified_at: datetime) -> Path:
    path = _content_path(root, content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    timestamp = modified_at.timestamp()
    os.utime(path, (timestamp, timestamp))
    return path


async def test_cleanup_orphans_is_grace_based_reference_safe_and_idempotent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "blobs"
    store = LocalBlobStore(root, max_blob_bytes=1024)
    first = await store.put(_put("blob-shared-first"), CONTENT)
    second = await store.put(_put("blob-shared-second"), CONTENT)
    assert first.digest == second.digest

    cutoff = datetime.now(timezone.utc)
    stale = cutoff - timedelta(hours=2)
    recent = cutoff + timedelta(hours=2)
    shared_path = _content_path(root)
    os.utime(shared_path, (stale.timestamp(), stale.timestamp()))

    temp_directory = root / "tmp"
    temp_directory.mkdir(parents=True, exist_ok=True)
    stale_temp = temp_directory / "put-stale.tmp"
    recent_temp = temp_directory / "put-recent.tmp"
    stale_temp.write_bytes(b"stale partial write")
    recent_temp.write_bytes(b"recent partial write")
    os.utime(stale_temp, (stale.timestamp(), stale.timestamp()))
    os.utime(recent_temp, (recent.timestamp(), recent.timestamp()))

    stale_orphan = _write_orphan(root, b"stale orphan", modified_at=stale)
    recent_orphan = _write_orphan(root, b"recent orphan", modified_at=recent)

    assert await store.cleanup_orphans(before=cutoff) == 2
    assert not stale_temp.exists()
    assert not stale_orphan.exists()
    assert recent_temp.read_bytes() == b"recent partial write"
    assert recent_orphan.read_bytes() == b"recent orphan"
    assert shared_path.read_bytes() == CONTENT
    assert await store.metadata("blob-shared-first") is not None
    assert await store.metadata("blob-shared-second") is not None

    assert await store.cleanup_orphans(before=cutoff) == 0
    assert recent_temp.exists()
    assert recent_orphan.exists()
    assert shared_path.exists()


async def test_cleanup_orphans_rejects_naive_cutoff(tmp_path: Path) -> None:
    store = LocalBlobStore(tmp_path / "blobs", max_blob_bytes=1024)

    with pytest.raises(ValueError, match="timezone-aware"):
        await store.cleanup_orphans(before=datetime(2026, 7, 18, 12, 30))


@pytest.mark.parametrize("candidate_kind", ("temporary", "content"))
async def test_cleanup_orphans_rejects_symlink_candidates_without_touching_target(
    tmp_path: Path,
    candidate_kind: str,
) -> None:
    root = tmp_path / "blobs"
    store = LocalBlobStore(root, max_blob_bytes=1024)
    outside = tmp_path / "outside-target"
    outside.write_bytes(b"must survive cleanup")

    if candidate_kind == "temporary":
        candidate = root / "tmp" / "stale-link.tmp"
    else:
        digest_hex = hashlib.sha256(b"symlink orphan").hexdigest()
        candidate = root / "sha256" / digest_hex[:2] / digest_hex
    candidate.parent.mkdir(parents=True, exist_ok=True)
    candidate.symlink_to(outside)

    with pytest.raises(BlobStoreError, match="symlink"):
        await store.cleanup_orphans(
            before=datetime.now(timezone.utc) + timedelta(days=1)
        )

    assert candidate.is_symlink()
    assert outside.read_bytes() == b"must survive cleanup"
