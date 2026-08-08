"""One concrete, bounded artifact store inside an admitted agent home."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import stat
import threading
from collections.abc import Callable
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import NoReturn, Protocol
from uuid import uuid4

from .._json import canonical_json
from ..capabilities import ArtifactPolicy
from .models import (
    MAX_ARTIFACT_BYTES_PER_AGENT,
    MAX_ARTIFACT_BYTES_PER_RUN,
    MAX_ARTIFACTS_PER_AGENT,
    MAX_ARTIFACTS_PER_RUN,
    ArtifactDraft,
    ArtifactError,
    ArtifactPayload,
    ArtifactRef,
    artifact_ref_from_mapping,
    artifact_ref_to_mapping,
    canonical_artifact_filename,
)

_RUN_ID = re.compile(r"run-[0-9a-f]{32}\Z")
_ARTIFACT_ID = re.compile(r"artifact-[0-9a-f]{32}\Z")
_STAGING_NAME = re.compile(r"artifact-[0-9a-f]{32}\.[0-9a-f]{32}\Z")
_CONFIG_STAGING_NAME = re.compile(r"delivery-config\.[0-9a-f]{32}\.tmp\Z")
_MAX_STAGING_ENTRIES = 1_024
_MAX_MANIFEST_BYTES = 64 * 1_024
_COMMIT_TIMEOUT_SECONDS = 30.0


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class ArtifactReferenceReader(Protocol):
    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]: ...


class _CancelledBeforePublication(BaseException):
    pass


class _PublicationGate:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._published = False

    def cancel(self) -> bool:
        with self._lock:
            self._cancelled = True
            return self._published

    def require_active(self) -> None:
        with self._lock:
            if self._cancelled:
                raise _CancelledBeforePublication

    def publish(self, action: Callable[[], None]) -> None:
        with self._lock:
            if self._cancelled:
                raise _CancelledBeforePublication
            action()
            self._published = True


class AgentHomeArtifactStore:
    """Commit and verify immutable payloads under one fixed agent-home layout."""

    def __init__(
        self,
        *,
        agent_id: str,
        agent_home: Path,
        references: ArtifactReferenceReader,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
        admission_error: ArtifactError | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_home = agent_home
        self.root = agent_home / "artifacts"
        self.staging = self.root / ".staging"
        self._references = references
        self._clock = clock
        self._id_factory = id_factory
        self._commit_lock = threading.Lock()
        self._admission_error = admission_error

    @classmethod
    async def open(
        cls,
        *,
        agent_id: str,
        agent_home: Path,
        references: ArtifactReferenceReader,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
    ) -> AgentHomeArtifactStore:
        store = cls(
            agent_id=agent_id,
            agent_home=agent_home,
            references=references,
            clock=clock,
            id_factory=id_factory,
        )
        try:
            refs = await references.list_artifact_refs(agent_id)
        except asyncio.CancelledError:
            raise
        except ArtifactError as error:
            store._admission_error = error
            return store
        except Exception as error:
            store._admission_error = ArtifactError(
                "artifact_storage_failed",
                "Artifact storage could not be admitted.",
                {"stage": "admission"},
            )
            store._admission_error.__cause__ = error
            return store

        worker = asyncio.create_task(asyncio.to_thread(store._admit_and_cleanup, refs))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
            except Exception:
                break
        try:
            worker.result()
        except ArtifactError as error:
            store._admission_error = error
        except Exception as error:
            store._admission_error = ArtifactError(
                "artifact_storage_failed",
                "Artifact storage could not be admitted.",
                {"stage": "admission"},
            )
            store._admission_error.__cause__ = error
        if cancelled:
            raise asyncio.CancelledError
        return store

    @property
    def available(self) -> bool:
        return self._admission_error is None

    async def close(self) -> None:
        return None

    async def list_refs(
        self,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        self._require_available()
        refs = await self._references.list_artifact_refs(
            self.agent_id,
            run_id=run_id,
            conversation_id=conversation_id,
        )
        return tuple(sorted(refs, key=lambda item: (item.created_at, item.artifact_id)))

    async def find_ref(self, artifact_id: str) -> ArtifactRef:
        self._require_available()
        if (
            not isinstance(artifact_id, str)
            or _ARTIFACT_ID.fullmatch(artifact_id) is None
        ):
            raise ArtifactError(
                "artifact_missing",
                "The requested artifact is not available.",
                {"artifact_id": str(artifact_id)},
            )
        return next(
            (
                item
                for item in await self.list_refs()
                if item.artifact_id == artifact_id
            ),
            None,
        ) or _raise_missing(artifact_id)

    async def read(self, artifact_id: str) -> ArtifactPayload:
        ref = await self.find_ref(artifact_id)
        return await asyncio.to_thread(self._read_ref, ref)

    async def read_ref(self, ref: ArtifactRef) -> ArtifactPayload:
        self._require_available()
        if ref not in await self.list_refs():
            raise ArtifactError(
                "artifact_missing",
                "The requested artifact is not available.",
                {"artifact_id": ref.artifact_id},
            )
        return await asyncio.to_thread(self._read_ref, ref)

    async def commit(
        self,
        draft: ArtifactDraft,
        policy: ArtifactPolicy,
        *,
        run_id: str,
        conversation_id: str,
        call_id: str,
        capability_id: str,
    ) -> ArtifactRef:
        self._require_available()
        gate = _PublicationGate()
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._commit_sync,
                draft,
                policy,
                run_id,
                conversation_id,
                call_id,
                capability_id,
                gate,
            )
        )
        cancelled = False
        try:
            async with asyncio.timeout(_COMMIT_TIMEOUT_SECONDS):
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        cancelled = True
                        gate.cancel()
        except TimeoutError:
            gate.cancel()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact commit exceeded its I/O limit.",
                {"stage": "commit_timeout"},
            )
        try:
            ref = worker.result()
        except _CancelledBeforePublication:
            raise asyncio.CancelledError from None
        if cancelled:
            raise asyncio.CancelledError
        return ref

    async def remove_all_run_artifacts(self) -> None:
        """Remove unreachable run directories while retaining delivery config."""

        self._require_available()
        await asyncio.to_thread(self._remove_all_run_artifacts_sync)

    def _require_available(self) -> None:
        if self._admission_error is not None:
            raise ArtifactError(
                self._admission_error.code,
                self._admission_error.message,
                self._admission_error.details,
            )

    def _admit_and_cleanup(self, refs: tuple[ArtifactRef, ...]) -> None:
        try:
            home = self.agent_home.resolve(strict=True)
            if not home.is_dir() or self.agent_home.is_symlink():
                raise OSError("agent home is not a contained directory")
            self.agent_home = home
            self.root = home / "artifacts"
            self.staging = self.root / ".staging"
            _mkdir_private(self.root)
            _mkdir_private(self.staging)
            with os.scandir(self.staging) as staging_iterator:
                staging_entries = tuple(staging_iterator)
            if len(staging_entries) > _MAX_STAGING_ENTRIES:
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact staging cleanup exceeds its admission bound.",
                    {"stage": "staging_cleanup"},
                )
            for entry in staging_entries:
                _remove_staging_entry(Path(entry.path))

            referenced = {item.artifact_id: item for item in refs}
            final_count = 0
            for run_entry in _run_entries(self.root):
                run_path = Path(run_entry.path)
                if run_entry.is_symlink() or not run_entry.is_dir(
                    follow_symlinks=False
                ):
                    raise OSError("artifact run entry is not a directory")
                if _RUN_ID.fullmatch(run_entry.name) is None:
                    raise OSError("artifact run entry has an invalid identity")
                with os.scandir(run_path) as artifact_iterator:
                    artifact_entries = tuple(artifact_iterator)
                for artifact_entry in artifact_entries:
                    final_count += 1
                    if final_count > MAX_ARTIFACTS_PER_AGENT:
                        raise ArtifactError(
                            "artifact_storage_failed",
                            "Artifact cleanup exceeds its final-directory bound.",
                            {"stage": "orphan_cleanup"},
                        )
                    artifact_path = Path(artifact_entry.path)
                    if _ARTIFACT_ID.fullmatch(artifact_entry.name) is None:
                        raise OSError("artifact directory has an invalid identity")
                    ref = referenced.get(artifact_entry.name)
                    if ref is None:
                        _remove_artifact_directory(artifact_path)
                        continue
                    if ref.run_id != run_entry.name:
                        raise OSError("referenced artifact is stored under another run")
                    # Referenced corruption is deliberately retained for explicit read errors.
                with os.scandir(run_path) as remaining:
                    empty = next(remaining, None) is None
                if empty:
                    run_path.rmdir()
            _fsync_directory(self.root)
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact storage admission or cleanup failed.",
                {"stage": "admission_cleanup"},
            ) from error

    def _commit_sync(
        self,
        draft: ArtifactDraft,
        policy: ArtifactPolicy,
        run_id: str,
        conversation_id: str,
        call_id: str,
        capability_id: str,
        gate: _PublicationGate,
    ) -> ArtifactRef:
        if not isinstance(draft, ArtifactDraft):
            raise TypeError("artifact commit requires ArtifactDraft")
        if not isinstance(policy, ArtifactPolicy):
            raise TypeError("artifact commit requires ArtifactPolicy")
        self._verify_storage_roots()
        if _RUN_ID.fullmatch(run_id) is None:
            raise ArtifactError(
                "artifact_storage_failed",
                "The artifact run identity is not path-safe.",
                {"stage": "identity"},
            )
        filename = canonical_artifact_filename(
            draft.suggested_filename,
            draft.media_type,
            policy.allowed_extensions,
        )
        if draft.media_type not in policy.allowed_media_types:
            raise ArtifactError(
                "artifact_invalid_format",
                "The artifact media type is not allowed.",
                {
                    "media_type": draft.media_type,
                    "allowed_extensions": (),
                },
            )
        size = len(draft.content)
        if (
            size > policy.max_bytes_per_artifact
            or size > policy.max_total_bytes_per_call
        ):
            raise ArtifactError(
                "artifact_quota_exceeded",
                "The artifact exceeds its capability byte limit.",
                {
                    "scope": "call",
                    "limit_kind": "bytes",
                    "limit": min(
                        policy.max_bytes_per_artifact,
                        policy.max_total_bytes_per_call,
                    ),
                    "attempted": size,
                },
            )
        with self._commit_lock:
            gate.require_active()
            run_count, run_bytes, agent_count, agent_bytes = self._quota_usage(run_id)
            _check_quota("run", "count", MAX_ARTIFACTS_PER_RUN, run_count + 1)
            _check_quota("run", "bytes", MAX_ARTIFACT_BYTES_PER_RUN, run_bytes + size)
            _check_quota("agent", "count", MAX_ARTIFACTS_PER_AGENT, agent_count + 1)
            _check_quota(
                "agent", "bytes", MAX_ARTIFACT_BYTES_PER_AGENT, agent_bytes + size
            )
            artifact_id = self._id_factory("artifact")
            if (
                not isinstance(artifact_id, str)
                or _ARTIFACT_ID.fullmatch(artifact_id) is None
            ):
                raise ArtifactError(
                    "artifact_storage_failed",
                    "The artifact identity factory returned an invalid identity.",
                    {"stage": "identity"},
                )
            digest = "sha256:" + sha256(draft.content).hexdigest()
            created_at = self._clock()
            ref = ArtifactRef(
                artifact_id=artifact_id,
                run_id=run_id,
                conversation_id=conversation_id,
                call_id=call_id,
                capability_id=capability_id,
                filename=filename,
                media_type=draft.media_type,
                byte_size=size,
                sha256=digest,
                sensitivity=draft.sensitivity,
                provenance=draft.provenance,
                created_at=created_at,
            )
            run_path = self.root / run_id
            final = run_path / artifact_id
            staging = self.staging / f"{artifact_id}.{uuid4().hex}"
            published = False
            try:
                _mkdir_private(staging, exclusive=True)
                gate.require_active()
                _write_exclusive(staging / "payload", draft.content)
                gate.require_active()
                manifest = canonical_json(artifact_ref_to_mapping(ref)).encode("utf-8")
                _write_exclusive(staging / "manifest.json", manifest)
                _fsync_directory(staging)
                gate.require_active()
                _mkdir_private(run_path)
                if final.exists() or final.is_symlink():
                    raise OSError(errno.EEXIST, "artifact identity collision")

                def publish() -> None:
                    nonlocal published
                    os.rename(staging, final)
                    published = True

                gate.publish(publish)
                _fsync_directory(final)
                _fsync_directory(run_path)
                _fsync_directory(self.root)
                return ref
            except _CancelledBeforePublication:
                if not published:
                    _remove_staging_entry(staging)
                raise
            except ArtifactError:
                if not published:
                    _remove_staging_entry(staging)
                raise
            except Exception as error:
                if not published:
                    _remove_staging_entry(staging)
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact commit failed.",
                    {"stage": "publish" if published else "staging"},
                ) from error

    def _quota_usage(self, run_id: str) -> tuple[int, int, int, int]:
        run_count = 0
        run_bytes = 0
        agent_count = 0
        agent_bytes = 0
        for run_entry in _run_entries(self.root):
            if (
                _RUN_ID.fullmatch(run_entry.name) is None
                or run_entry.is_symlink()
                or not run_entry.is_dir(follow_symlinks=False)
            ):
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact quota accounting found an invalid entry.",
                    {"stage": "quota"},
                )
            with os.scandir(run_entry.path) as artifact_iterator:
                artifact_entries = tuple(artifact_iterator)
            for artifact_entry in artifact_entries:
                if (
                    _ARTIFACT_ID.fullmatch(artifact_entry.name) is None
                    or artifact_entry.is_symlink()
                    or not artifact_entry.is_dir(follow_symlinks=False)
                ):
                    raise ArtifactError(
                        "artifact_storage_failed",
                        "Artifact quota accounting found an invalid artifact entry.",
                        {"stage": "quota"},
                    )
                agent_count += 1
                if agent_count > MAX_ARTIFACTS_PER_AGENT:
                    return run_count, run_bytes, agent_count, agent_bytes
                artifact_path = Path(artifact_entry.path)
                payload = artifact_path / "payload"
                try:
                    facts = payload.lstat()
                except OSError as error:
                    raise ArtifactError(
                        "artifact_storage_failed",
                        "Artifact quota accounting found a missing payload.",
                        {"stage": "quota"},
                    ) from error
                if not stat.S_ISREG(facts.st_mode) or payload.is_symlink():
                    raise ArtifactError(
                        "artifact_storage_failed",
                        "Artifact quota accounting found an invalid payload.",
                        {"stage": "quota"},
                    )
                payload_size = facts.st_size
                agent_bytes += payload_size
                if run_entry.name == run_id:
                    run_count += 1
                    run_bytes += payload_size
        return run_count, run_bytes, agent_count, agent_bytes

    def _read_ref(self, ref: ArtifactRef) -> ArtifactPayload:
        try:
            self._verify_storage_roots()
        except ArtifactError:
            _corrupt(ref.artifact_id, "storage_root_changed")
        if (
            _RUN_ID.fullmatch(ref.run_id) is None
            or _ARTIFACT_ID.fullmatch(ref.artifact_id) is None
        ):
            raise ArtifactError(
                "artifact_missing",
                "The requested artifact is not available.",
                {"artifact_id": ref.artifact_id},
            )
        directory = self.root / ref.run_id / ref.artifact_id
        try:
            facts = directory.lstat()
        except FileNotFoundError:
            raise ArtifactError(
                "artifact_missing",
                "The requested artifact is not available.",
                {"artifact_id": ref.artifact_id},
            ) from None
        if not stat.S_ISDIR(facts.st_mode) or directory.is_symlink():
            _corrupt(ref.artifact_id, "invalid_directory_type")
        try:
            resolved = directory.resolve(strict=True)
            resolved.relative_to(self.root.resolve(strict=True))
        except (OSError, ValueError):
            _corrupt(ref.artifact_id, "outside_agent_home")
        manifest = _read_regular(directory / "manifest.json", _MAX_MANIFEST_BYTES)
        try:
            raw_manifest = json.loads(manifest.decode("utf-8"))
            if not isinstance(raw_manifest, dict):
                raise ValueError("manifest is not an object")
            stored_ref = artifact_ref_from_mapping(raw_manifest)
        except Exception:
            _corrupt(ref.artifact_id, "malformed_manifest")
        if stored_ref != ref or manifest != canonical_json(
            artifact_ref_to_mapping(ref)
        ).encode("utf-8"):
            _corrupt(ref.artifact_id, "manifest_mismatch")
        content = _read_regular(directory / "payload", ref.byte_size)
        if len(content) != ref.byte_size:
            _corrupt(ref.artifact_id, "size_mismatch")
        digest = "sha256:" + sha256(content).hexdigest()
        if digest != ref.sha256:
            _corrupt(ref.artifact_id, "digest_mismatch")
        return ArtifactPayload(ref=ref, content=content)

    def _remove_all_run_artifacts_sync(self) -> None:
        self._verify_storage_roots()
        count = 0
        for entry in _run_entries(self.root):
            if (
                _RUN_ID.fullmatch(entry.name) is None
                or entry.is_symlink()
                or not entry.is_dir(follow_symlinks=False)
            ):
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact conversation cleanup found an invalid entry.",
                    {"stage": "conversation_clear"},
                )
            with os.scandir(entry.path) as artifact_entries:
                artifacts = tuple(artifact_entries)
            count += len(artifacts)
            if count > MAX_ARTIFACTS_PER_AGENT:
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact conversation cleanup exceeds its bound.",
                    {"stage": "conversation_clear"},
                )
            for artifact_entry in artifacts:
                if _ARTIFACT_ID.fullmatch(artifact_entry.name) is None:
                    raise ArtifactError(
                        "artifact_storage_failed",
                        "Artifact conversation cleanup found an invalid identity.",
                        {"stage": "conversation_clear"},
                    )
                _remove_artifact_directory(Path(artifact_entry.path))
            Path(entry.path).rmdir()
        _fsync_directory(self.root)

    def _verify_storage_roots(self) -> None:
        try:
            home = self.agent_home.resolve(strict=True)
            root_facts = self.root.lstat()
            staging_facts = self.staging.lstat()
            root = self.root.resolve(strict=True)
            staging = self.staging.resolve(strict=True)
        except OSError as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact storage identity is unavailable.",
                {"stage": "containment"},
            ) from error
        if (
            self.agent_home.is_symlink()
            or self.root.is_symlink()
            or self.staging.is_symlink()
            or not stat.S_ISDIR(root_facts.st_mode)
            or not stat.S_ISDIR(staging_facts.st_mode)
            or root.parent != home
            or staging.parent != root
        ):
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact storage identity changed.",
                {"stage": "containment"},
            )


def _raise_missing(artifact_id: str) -> NoReturn:
    raise ArtifactError(
        "artifact_missing",
        "The requested artifact is not available.",
        {"artifact_id": artifact_id},
    )


def _check_quota(scope: str, kind: str, limit: int, attempted: int) -> None:
    if attempted > limit:
        raise ArtifactError(
            "artifact_quota_exceeded",
            "The artifact quota would be exceeded.",
            {
                "scope": scope,
                "limit_kind": kind,
                "limit": limit,
                "attempted": attempted,
            },
        )


def _run_entries(root: Path) -> tuple[os.DirEntry[str], ...]:
    with os.scandir(root) as entries:
        return tuple(
            entry
            for entry in entries
            if entry.name not in {".staging", "delivery-config.json"}
        )


def _mkdir_private(path: Path, *, exclusive: bool = False) -> None:
    if path.exists() or path.is_symlink():
        facts = path.lstat()
        if exclusive or not stat.S_ISDIR(facts.st_mode) or path.is_symlink():
            raise OSError(errno.EEXIST, "private directory already exists")
        os.chmod(path, 0o700)
        return
    path.mkdir(mode=0o700, parents=not exclusive, exist_ok=not exclusive)
    os.chmod(path, 0o700)


def _write_exclusive(path: Path, content: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(content)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise OSError("artifact write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o600)
    finally:
        os.close(descriptor)


def _read_regular(path: Path, maximum: int) -> bytes:
    try:
        facts = path.lstat()
    except FileNotFoundError:
        _corrupt(path.parent.name, "missing_entry")
    if not stat.S_ISREG(facts.st_mode) or path.is_symlink():
        _corrupt(path.parent.name, "invalid_entry_type")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        _corrupt(path.parent.name, "invalid_entry_type")
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            _corrupt(path.parent.name, "invalid_entry_type")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        content = b"".join(chunks)
        if len(content) > maximum:
            _corrupt(path.parent.name, "size_mismatch")
        return content
    finally:
        os.close(descriptor)


def _corrupt(artifact_id: str, reason: str) -> NoReturn:
    raise ArtifactError(
        "artifact_corrupt",
        "The stored artifact failed integrity verification.",
        {"artifact_id": artifact_id, "reason": reason},
    )


def _remove_staging_entry(path: Path) -> None:
    try:
        facts = path.lstat()
    except FileNotFoundError:
        return
    if _CONFIG_STAGING_NAME.fullmatch(path.name) is not None:
        if not stat.S_ISREG(facts.st_mode) or path.is_symlink():
            raise OSError("artifact config staging entry has an invalid type")
        path.unlink()
        return
    if _STAGING_NAME.fullmatch(path.name) is None:
        raise OSError("artifact staging entry has an invalid identity")
    _remove_artifact_directory(path)


def _remove_artifact_directory(path: Path) -> None:
    facts = path.lstat()
    if not stat.S_ISDIR(facts.st_mode) or path.is_symlink():
        raise OSError("artifact cleanup entry is not an exact directory")
    with os.scandir(path) as entries:
        children = tuple(entries)
    if len(children) > 2 or any(
        child.name not in {"manifest.json", "payload"}
        or child.is_symlink()
        or not child.is_file(follow_symlinks=False)
        for child in children
    ):
        raise OSError("artifact cleanup entry exceeds its fixed shape")
    for child in children:
        Path(child.path).unlink()
    path.rmdir()


def _fsync_directory(path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_DIRECTORY | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = ["AgentHomeArtifactStore", "ArtifactReferenceReader"]
