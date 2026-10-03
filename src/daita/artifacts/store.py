"""Commit, read, list, and clean bounded artifacts within an admitted agent home."""

from __future__ import annotations

import asyncio
import errno
import json
import os
import re
import stat
import threading
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import NoReturn, Protocol, cast
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
    ArtifactRecord,
    ArtifactRef,
    ArtifactState,
    artifact_provenance_to_mapping,
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
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class ArtifactRegistry(Protocol):
    async def get_artifact_record(self, artifact_id: str) -> ArtifactRecord | None: ...
    async def list_artifact_records(
        self,
        agent_id: str,
        *,
        state: ArtifactState | None = None,
        run_id: str | None = None,
        conversation_id: str | None = None,
        caller_principal_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[ArtifactRecord, ...]: ...
    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
        caller_principal_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[ArtifactRef, ...]: ...
    async def begin_artifact_creation(
        self, record: ArtifactRecord, *, reserved: bool = False
    ) -> None: ...
    async def transition_artifact(
        self, record: ArtifactRecord, state: ArtifactState
    ) -> ArtifactRecord: ...
    async def finish_artifact_deletion(self, record: ArtifactRecord) -> None: ...


async def _drain(operation: Awaitable[object]) -> None:
    worker = asyncio.ensure_future(operation)
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
    worker.result()
    if cancelled:
        raise asyncio.CancelledError


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
        registry: ArtifactRegistry,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
        admission_error: ArtifactError | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_home = agent_home
        self.root = agent_home / "artifacts"
        self.staging = self.root / ".staging"
        self._registry = registry
        self._clock = clock
        self._id_factory = id_factory
        self._commit_lock = threading.Lock()
        self._lifecycle_lock = asyncio.Lock()
        self._admission_error = admission_error

    @classmethod
    async def open(
        cls,
        *,
        agent_id: str,
        agent_home: Path,
        registry: ArtifactRegistry,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
    ) -> AgentHomeArtifactStore:
        store = cls(
            agent_id=agent_id,
            agent_home=agent_home,
            registry=registry,
            clock=clock,
            id_factory=id_factory,
        )
        try:
            records = await registry.list_artifact_records(agent_id)
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

        worker = asyncio.create_task(
            asyncio.to_thread(
                store._admit_and_cleanup,
                records,
            )
        )
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
        if store.available:
            try:
                await _drain(store._recover_pending(records))
            except asyncio.CancelledError:
                raise
            except ArtifactError as error:
                store._admission_error = error
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
        caller_principal_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> tuple[ArtifactRef, ...]:
        async with self._lifecycle_lock:
            self._require_available()
            return await self._registry.list_artifact_refs(
                self.agent_id,
                run_id=run_id,
                conversation_id=conversation_id,
                caller_principal_id=caller_principal_id,
                limit=limit,
                offset=offset,
            )

    async def find_ref(self, artifact_id: str) -> ArtifactRef:
        async with self._lifecycle_lock:
            return await self._find_ref(artifact_id)

    async def _find_ref(self, artifact_id: str) -> ArtifactRef:
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
        record = await self._registry.get_artifact_record(artifact_id)
        if (
            record is None
            or record.agent_id != self.agent_id
            or record.state is not ArtifactState.READY
        ):
            _raise_missing(artifact_id)
        return record.ref

    async def read(self, artifact_id: str) -> ArtifactPayload:
        async with self._lifecycle_lock:
            ref = await self._find_ref(artifact_id)
            return await asyncio.to_thread(self._read_ref, ref)

    async def read_ref(self, ref: ArtifactRef) -> ArtifactPayload:
        async with self._lifecycle_lock:
            self._require_available()
            if await self._find_ref(ref.artifact_id) != ref:
                _raise_missing(ref.artifact_id)
            return await asyncio.to_thread(self._read_ref, ref)

    async def delete(self, artifact_id: str) -> bool:
        """Remove files and their lifecycle row, retaining only unfinished cleanup."""
        async with self._lifecycle_lock:
            self._require_available()
            record = await self._registry.get_artifact_record(artifact_id)
            if record is None:
                return False
            if (
                record.agent_id != self.agent_id
                or record.state is ArtifactState.CREATING
            ):
                _raise_missing(artifact_id)
            newly_deleted = record.state is ArtifactState.READY
            if newly_deleted:
                record = await self._registry.transition_artifact(
                    record, ArtifactState.DELETING
                )
            await _drain(self._finish_cleanup(record))
            return newly_deleted

    async def _finish_cleanup(self, record: ArtifactRecord) -> None:
        try:
            await asyncio.to_thread(self._delete_sync, record)
            await self._registry.finish_artifact_deletion(record)
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact cleanup is incomplete. Retry deletion.",
                {"artifact_id": record.ref.artifact_id, "stage": "delete_cleanup"},
            ) from error

    async def _recover_pending(self, records: tuple[ArtifactRecord, ...]) -> None:
        for record in records:
            if record.state is ArtifactState.DELETING:
                await self._finish_cleanup(record)
            elif record.state is ArtifactState.CREATING:
                await self._recover_creation(record)

    async def _recover_creation(self, record: ArtifactRecord) -> ArtifactRef | None:
        ref = await asyncio.to_thread(
            self._recover_reserved_sync, record.ref.run_id, record.ref.artifact_id
        )
        if ref is not None:
            if ref != record.ref:
                _corrupt(ref.artifact_id, "creating_manifest_mismatch")
            await self._registry.transition_artifact(record, ArtifactState.READY)
            return ref
        pending = await self._registry.transition_artifact(
            record, ArtifactState.DELETING
        )
        await self._finish_cleanup(pending)
        return None

    def _delete_sync(self, record: ArtifactRecord) -> None:
        try:
            self._verify_storage_roots()
            run_path = self.root / record.ref.run_id
            try:
                facts = run_path.lstat()
            except FileNotFoundError:
                _fsync_directory(self.root)
                return
            if not stat.S_ISDIR(facts.st_mode) or run_path.is_symlink():
                raise OSError("artifact run entry is not an exact directory")
            with self._commit_lock:
                path = run_path / record.ref.artifact_id
                try:
                    path.lstat()
                except FileNotFoundError:
                    pass
                else:
                    _remove_artifact_directory(path)
                _fsync_directory(run_path)
        except Exception as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "The artifact is deleted, but file cleanup is incomplete. Retry deletion.",
                {"artifact_id": record.ref.artifact_id, "stage": "delete_cleanup"},
            ) from error

    async def recover_reserved(
        self,
        run_id: str,
        artifact_id: str,
    ) -> ArtifactRef | None:
        """Recover one exactly reserved published artifact after host loss."""

        async with self._lifecycle_lock:
            return await self._recover_reserved(run_id, artifact_id)

    async def _recover_reserved(
        self, run_id: str, artifact_id: str
    ) -> ArtifactRef | None:
        self._require_available()
        record = await self._registry.get_artifact_record(artifact_id)
        if (
            record is None
            or record.agent_id != self.agent_id
            or record.ref.run_id != run_id
            or record.state is ArtifactState.DELETING
        ):
            return None
        if record.state is ArtifactState.CREATING:
            return await self._recover_creation(record)
        await asyncio.to_thread(self._read_ref, record.ref)
        return record.ref

    async def read_reserved(
        self,
        run_id: str,
        artifact_id: str,
    ) -> ArtifactPayload | None:
        """Read one restart-reserved artifact before its durable result promotion."""

        async with self._lifecycle_lock:
            ref = await self._recover_reserved(run_id, artifact_id)
            if ref is None:
                return None
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
        reserved_artifact_id: str | None = None,
        caller_principal_id: str | None = None,
    ) -> ArtifactRef:
        async with self._lifecycle_lock:
            self._require_available()
            artifact_id = reserved_artifact_id or self._id_factory("artifact")
            if (
                not isinstance(artifact_id, str)
                or _ARTIFACT_ID.fullmatch(artifact_id) is None
            ):
                raise ArtifactError(
                    "artifact_storage_failed",
                    "The artifact identity is invalid.",
                    {"stage": "identity"},
                )
            ref = self._prepare_ref(
                draft,
                policy,
                run_id,
                conversation_id,
                call_id,
                capability_id,
                artifact_id,
            )
            record = ArtifactRecord(
                ref,
                self.agent_id,
                caller_principal_id or self.agent_id,
                ArtifactState.CREATING,
            )
            await self._registry.begin_artifact_creation(
                record, reserved=reserved_artifact_id is not None
            )
            try:
                return await self._commit(draft, ref)
            finally:
                await _drain(self._recover_creation(record))

    async def _commit(self, draft: ArtifactDraft, ref: ArtifactRef) -> ArtifactRef:
        self._require_available()
        gate = _PublicationGate()
        worker = asyncio.create_task(
            asyncio.to_thread(
                self._commit_sync,
                draft,
                ref,
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

    def _require_available(self) -> None:
        if self._admission_error is not None:
            raise ArtifactError(
                self._admission_error.code,
                self._admission_error.message,
                self._admission_error.details,
            )

    def _admit_and_cleanup(
        self,
        records: tuple[ArtifactRecord, ...],
    ) -> None:
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

            referenced = {item.ref.artifact_id: item.ref for item in records}
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

    def _recover_reserved_sync(
        self,
        run_id: str,
        artifact_id: str,
    ) -> ArtifactRef | None:
        if (
            _RUN_ID.fullmatch(run_id) is None
            or _ARTIFACT_ID.fullmatch(artifact_id) is None
        ):
            raise ArtifactError(
                "artifact_storage_failed",
                "The reserved artifact identity is invalid.",
                {"stage": "reservation_identity"},
            )
        directory = self.root / run_id / artifact_id
        if not directory.exists():
            return None
        try:
            ref = self._load_manifest_ref(run_id, artifact_id)
            self._read_ref(ref)
            _fsync_directory(directory)
            _fsync_directory(directory.parent)
            _fsync_directory(self.root)
            return ref
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactError(
                "artifact_corrupt",
                "The reserved job artifact could not be reconciled safely.",
                {"artifact_id": artifact_id},
            ) from error

    def _prepare_ref(
        self,
        draft: ArtifactDraft,
        policy: ArtifactPolicy,
        run_id: str,
        conversation_id: str,
        call_id: str,
        capability_id: str,
        artifact_id: str,
    ) -> ArtifactRef:
        if not isinstance(draft, ArtifactDraft):
            raise TypeError("artifact commit requires ArtifactDraft")
        if not isinstance(policy, ArtifactPolicy):
            raise TypeError("artifact commit requires ArtifactPolicy")
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
        return ArtifactRef(
            artifact_id=artifact_id,
            run_id=run_id,
            conversation_id=conversation_id,
            call_id=call_id,
            capability_id=capability_id,
            filename=filename,
            media_type=draft.media_type,
            byte_size=size,
            sha256="sha256:" + sha256(draft.content).hexdigest(),
            sensitivity=draft.sensitivity,
            provenance=draft.provenance,
            created_at=self._clock(),
        )

    def _commit_sync(
        self, draft: ArtifactDraft, ref: ArtifactRef, gate: _PublicationGate
    ) -> ArtifactRef:
        self._verify_storage_roots()
        artifact_id, run_id = ref.artifact_id, ref.run_id
        with self._commit_lock:
            gate.require_active()
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

    def _read_ref(self, ref: ArtifactRef) -> ArtifactPayload:
        stored_ref = self._load_manifest_ref(ref.run_id, ref.artifact_id)
        if stored_ref != ref:
            _corrupt(ref.artifact_id, "manifest_mismatch")
        directory = self.root / ref.run_id / ref.artifact_id
        content = _read_regular(directory / "payload", ref.byte_size)
        if len(content) != ref.byte_size:
            _corrupt(ref.artifact_id, "size_mismatch")
        digest = "sha256:" + sha256(content).hexdigest()
        if digest != ref.sha256:
            _corrupt(ref.artifact_id, "digest_mismatch")
        return ArtifactPayload(ref=ref, content=content)

    def _load_manifest_ref(self, run_id: str, artifact_id: str) -> ArtifactRef:
        try:
            self._verify_storage_roots()
        except ArtifactError:
            _corrupt(artifact_id, "storage_root_changed")
        if (
            _RUN_ID.fullmatch(run_id) is None
            or _ARTIFACT_ID.fullmatch(artifact_id) is None
        ):
            raise ArtifactError(
                "artifact_missing",
                "The requested artifact is not available.",
                {"artifact_id": artifact_id},
            )
        directory = self.root / run_id / artifact_id
        try:
            facts = directory.lstat()
        except FileNotFoundError:
            raise ArtifactError(
                "artifact_missing",
                "The requested artifact is not available.",
                {"artifact_id": artifact_id},
            ) from None
        if not stat.S_ISDIR(facts.st_mode) or directory.is_symlink():
            _corrupt(artifact_id, "invalid_directory_type")
        try:
            resolved = directory.resolve(strict=True)
            resolved.relative_to(self.root.resolve(strict=True))
        except (OSError, ValueError):
            _corrupt(artifact_id, "outside_agent_home")
        manifest = _read_regular(directory / "manifest.json", _MAX_MANIFEST_BYTES)
        try:
            raw_manifest = json.loads(manifest.decode("utf-8"))
            if not isinstance(raw_manifest, dict):
                raise ValueError("manifest is not an object")
            stored_ref = artifact_ref_from_mapping(raw_manifest)
        except Exception:
            _corrupt(artifact_id, "malformed_manifest")
        if (
            stored_ref.run_id != run_id
            or stored_ref.artifact_id != artifact_id
            or manifest
            != canonical_json(artifact_ref_to_mapping(stored_ref)).encode("utf-8")
        ):
            _corrupt(artifact_id, "manifest_mismatch")
        return stored_ref

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
    _fsync_directory(path.parent)


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


def validate_artifact_home(
    *,
    agent_id: str,
    agent_home: Path,
    records: tuple[ArtifactRecord, ...],
) -> None:
    """Validate the complete current artifact tree without cleaning or changing it."""

    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be non-empty text")
    home = Path(os.path.abspath(os.fspath(agent_home)))
    try:
        home_state = home.lstat()
    except OSError as error:
        raise ArtifactError(
            "artifact_storage_failed",
            "Artifact storage identity is unavailable.",
            {"stage": "home_validation"},
        ) from error
    if (
        not stat.S_ISDIR(home_state.st_mode)
        or stat.S_ISLNK(home_state.st_mode)
        or home.resolve(strict=True) != home
    ):
        raise ArtifactError(
            "artifact_storage_failed",
            "Artifact storage identity is invalid.",
            {"stage": "home_validation"},
        )
    root = home / "artifacts"
    if not root.exists():
        if any(record.state is ArtifactState.READY for record in records):
            raise ArtifactError(
                "artifact_corrupt",
                "Referenced artifact storage is missing.",
                {"artifact_id": "unknown"},
            )
        return
    root_state = root.lstat()
    if not stat.S_ISDIR(root_state.st_mode) or root.is_symlink():
        raise ArtifactError(
            "artifact_storage_failed",
            "Artifact storage root is invalid.",
            {"stage": "home_validation"},
        )
    staging = root / ".staging"
    staging_state = staging.lstat()
    if not stat.S_ISDIR(staging_state.st_mode) or staging.is_symlink():
        raise ArtifactError(
            "artifact_storage_failed",
            "Artifact staging root is invalid.",
            {"stage": "home_validation"},
        )
    with os.scandir(staging) as iterator:
        staging_entries = tuple(iterator)
    if len(staging_entries) > _MAX_STAGING_ENTRIES:
        raise ArtifactError(
            "artifact_storage_failed",
            "Artifact staging exceeds its fixed bound.",
            {"stage": "home_validation"},
        )
    for entry in staging_entries:
        path = Path(entry.path)
        facts = path.lstat()
        if _CONFIG_STAGING_NAME.fullmatch(entry.name) is not None:
            if not stat.S_ISREG(facts.st_mode) or path.is_symlink():
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact staging entry is invalid.",
                    {"stage": "home_validation"},
                )
            continue
        if (
            _STAGING_NAME.fullmatch(entry.name) is None
            or not stat.S_ISDIR(facts.st_mode)
            or path.is_symlink()
        ):
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact staging entry is invalid.",
                {"stage": "home_validation"},
            )
        with os.scandir(path) as children:
            staged_children = tuple(children)
        if len(staged_children) > 2 or any(
            child.name not in {"manifest.json", "payload"}
            or child.is_symlink()
            or not child.is_file(follow_symlinks=False)
            for child in staged_children
        ):
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact staging entry exceeds its fixed shape.",
                {"stage": "home_validation"},
            )

    store = AgentHomeArtifactStore(
        agent_id=agent_id,
        agent_home=home,
        registry=cast(ArtifactRegistry, object()),
    )
    pending = {
        (record.ref.run_id, record.ref.artifact_id)
        for record in records
        if record.state is not ArtifactState.READY
    }
    stored: dict[str, ArtifactRef] = {}
    count = 0
    byte_total = 0
    for run_entry in _run_entries(root):
        if (
            _RUN_ID.fullmatch(run_entry.name) is None
            or run_entry.is_symlink()
            or not run_entry.is_dir(follow_symlinks=False)
        ):
            raise ArtifactError(
                "artifact_storage_failed",
                "Artifact run entry is invalid.",
                {"stage": "home_validation"},
            )
        run_count = 0
        run_bytes = 0
        with os.scandir(run_entry.path) as iterator:
            entries = tuple(iterator)
        for entry in entries:
            if (
                _ARTIFACT_ID.fullmatch(entry.name) is None
                or entry.is_symlink()
                or not entry.is_dir(follow_symlinks=False)
            ):
                raise ArtifactError(
                    "artifact_storage_failed",
                    "Artifact entry is invalid.",
                    {"stage": "home_validation"},
                )
            if (run_entry.name, entry.name) in pending:
                # Interrupted deletion can leave either fixed file absent. Still
                # enforce shape and quotas before startup removes the directory.
                with os.scandir(entry.path) as children:
                    remaining = tuple(children)
                if len(remaining) > 2 or any(
                    child.name not in {"payload", "manifest.json"}
                    or child.is_symlink()
                    or not child.is_file(follow_symlinks=False)
                    for child in remaining
                ):
                    raise ArtifactError(
                        "artifact_storage_failed",
                        "Deleted artifact cleanup shape is invalid.",
                        {"stage": "home_validation"},
                    )
                size = sum(
                    child.stat(follow_symlinks=False).st_size
                    for child in remaining
                    if child.name == "payload"
                )
                count += 1
                run_count += 1
                byte_total += size
                run_bytes += size
                continue
            ref = store._load_manifest_ref(run_entry.name, entry.name)
            store._read_ref(ref)
            if ref.artifact_id in stored:
                raise ArtifactError(
                    "artifact_corrupt",
                    "Artifact identity is duplicated.",
                    {"artifact_id": ref.artifact_id},
                )
            stored[ref.artifact_id] = ref
            count += 1
            run_count += 1
            byte_total += ref.byte_size
            run_bytes += ref.byte_size
        _check_quota("run", "count", MAX_ARTIFACTS_PER_RUN, run_count)
        _check_quota("run", "bytes", MAX_ARTIFACT_BYTES_PER_RUN, run_bytes)
    _check_quota("agent", "count", MAX_ARTIFACTS_PER_AGENT, count)
    _check_quota("agent", "bytes", MAX_ARTIFACT_BYTES_PER_AGENT, byte_total)

    for record in records:
        if record.agent_id != agent_id:
            _corrupt(record.ref.artifact_id, "registry_owner_mismatch")
        if (
            record.state is ArtifactState.READY
            and stored.get(record.ref.artifact_id) != record.ref
        ):
            _corrupt(record.ref.artifact_id, "registered_manifest_mismatch")


__all__ = [
    "AgentHomeArtifactStore",
    "ArtifactRegistry",
    "validate_artifact_home",
]
