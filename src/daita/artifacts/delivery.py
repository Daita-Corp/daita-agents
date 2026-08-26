"""Authorize and publish committed artifacts to configured local destinations."""

from __future__ import annotations

import asyncio
import ctypes
import json
import os
import re
import stat
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import NoReturn, Protocol
from uuid import uuid4

from .._json import FrozenJsonObject, canonical_json
from ..adapters.models import SourceRegistration
from .models import (
    DEFAULT_DESTINATION_SELECTOR,
    MAX_COLLISION_SUFFIX,
    MAX_ONE_TIME_DESTINATIONS,
    MAX_PERSISTENT_DESTINATIONS,
    SYSTEM_DOWNLOADS_DESTINATION_ID,
    ArtifactDeliveryReceipt,
    ArtifactDestination,
    ArtifactDestinationKind,
    ArtifactError,
    ArtifactPayload,
    ArtifactRef,
    DestinationAuthorization,
    DestinationAvailability,
    canonical_artifact_filename,
)
from .store import AgentHomeArtifactStore

_DESTINATION_ID = re.compile(r"destination-[0-9a-f]{32}\Z")
_MAX_CONFIG_BYTES = 256 * 1024
_DELIVERY_TIMEOUT_SECONDS = 60.0


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


class DeliverySourceReader(Protocol):
    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]: ...


@dataclass(frozen=True, slots=True)
class _DestinationGrant:
    destination_id: str
    display_name: str
    kind: ArtifactDestinationKind
    authorization: DestinationAuthorization
    path: Path
    device: int
    inode: int
    grant_digest: str
    authorized_at: datetime
    run_id: str | None = None


class _CancelledBeforePublication(BaseException):
    pass


class _DeliveryGate:
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


class LocalArtifactDelivery:
    """Resolve exact grants and copy only verified committed artifacts."""

    def __init__(
        self,
        *,
        agent_id: str,
        agent_home: Path,
        artifacts: AgentHomeArtifactStore,
        sources: DeliverySourceReader,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
    ) -> None:
        self.agent_id = agent_id
        self.agent_home = agent_home.resolve(strict=True)
        self.artifacts = artifacts
        self._sources = sources
        self._clock = clock
        self._id_factory = id_factory
        self._config_path = self.agent_home / "artifacts" / "delivery-config.json"
        self._persistent: dict[str, _DestinationGrant] = {}
        self._one_time: dict[str, _DestinationGrant] = {}
        self._default_id: str | None = None
        self._system: _DestinationGrant | None = None
        self._config_error: ArtifactError | None = None
        self._downloads_error: ArtifactError | None = None

    @classmethod
    async def open(
        cls,
        *,
        agent_id: str,
        agent_home: Path,
        artifacts: AgentHomeArtifactStore,
        sources: DeliverySourceReader,
        downloads_directory: Path | None,
        clock: Callable[[], datetime] = _utc_now,
        id_factory: Callable[[str], str] = _new_id,
    ) -> LocalArtifactDelivery:
        owner = cls(
            agent_id=agent_id,
            agent_home=agent_home,
            artifacts=artifacts,
            sources=sources,
            clock=clock,
            id_factory=id_factory,
        )
        try:
            await asyncio.to_thread(owner._load_config)
        except ArtifactError as error:
            owner._config_error = error
        try:
            candidate = (
                downloads_directory
                if downloads_directory is not None
                else await asyncio.to_thread(resolve_os_downloads_directory)
            )
            owner._system = await owner._admit_path(
                candidate,
                destination_id=SYSTEM_DOWNLOADS_DESTINATION_ID,
                authorization=DestinationAuthorization.SYSTEM,
                kind=ArtifactDestinationKind.SYSTEM_DOWNLOADS,
                display_name="Downloads",
                run_id=None,
            )
        except asyncio.CancelledError:
            raise
        except ArtifactError as error:
            owner._downloads_error = ArtifactError(
                "artifact_downloads_unavailable",
                "The operating system Downloads folder is unavailable.",
                {},
            )
            owner._downloads_error.__cause__ = error
        except Exception as error:
            owner._downloads_error = ArtifactError(
                "artifact_downloads_unavailable",
                "The operating system Downloads folder is unavailable.",
                {},
            )
            owner._downloads_error.__cause__ = error
        return owner

    async def close(self) -> None:
        self._one_time.clear()

    async def model_destinations(self, run_id: str) -> tuple[ArtifactDestination, ...]:
        views: list[ArtifactDestination] = [self._system_view()]
        for grant in sorted(
            self._persistent.values(), key=lambda item: item.destination_id
        ):
            views.append(
                self._view(grant, is_default=grant.destination_id == self._default_id)
            )
        for grant in sorted(
            self._one_time.values(), key=lambda item: item.destination_id
        ):
            if grant.run_id == run_id:
                views.append(self._view(grant, is_default=False))
        return tuple(views)

    async def register_one_time(
        self, directory: Path, *, run_id: str
    ) -> ArtifactDestination:
        matching = sum(grant.run_id == run_id for grant in self._one_time.values())
        if matching >= MAX_ONE_TIME_DESTINATIONS:
            raise ArtifactError(
                "artifact_quota_exceeded",
                "The one-time destination limit would be exceeded.",
                {
                    "scope": "run",
                    "limit_kind": "one_time_destinations",
                    "limit": MAX_ONE_TIME_DESTINATIONS,
                    "attempted": matching + 1,
                },
            )
        destination_id = self._next_destination_id()
        grant = await self._admit_path(
            directory,
            destination_id=destination_id,
            authorization=DestinationAuthorization.ONE_TIME,
            kind=ArtifactDestinationKind.LOCAL_DIRECTORY,
            display_name=None,
            run_id=run_id,
        )
        self._one_time[destination_id] = grant
        return self._view(grant, is_default=False)

    def end_run(self, run_id: str) -> None:
        for destination_id in tuple(self._one_time):
            if self._one_time[destination_id].run_id == run_id:
                del self._one_time[destination_id]

    async def export_destination(self) -> ArtifactDestination:
        if self._default_id is not None:
            grant = self._persistent.get(self._default_id)
            if grant is None:
                raise ArtifactError(
                    "artifact_storage_failed",
                    "The persisted export-location configuration is invalid.",
                    {"stage": "delivery_config"},
                )
            return self._view(grant, is_default=True)
        return self._system_view()

    async def set_export_destination(self, directory: Path) -> ArtifactDestination:
        self._require_config()
        try:
            resolved = directory.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise ArtifactError(
                "artifact_destination_unavailable",
                "The artifact destination is unavailable.",
                {},
            ) from error
        existing = next(
            (item for item in self._persistent.values() if item.path == resolved),
            None,
        )
        created = False
        if existing is None:
            if len(self._persistent) >= MAX_PERSISTENT_DESTINATIONS:
                raise ArtifactError(
                    "artifact_quota_exceeded",
                    "The persistent destination limit would be exceeded.",
                    {
                        "scope": "agent",
                        "limit_kind": "persistent_destinations",
                        "limit": MAX_PERSISTENT_DESTINATIONS,
                        "attempted": len(self._persistent) + 1,
                    },
                )
            existing = await self._admit_path(
                directory,
                destination_id=self._next_destination_id(),
                authorization=DestinationAuthorization.PERSISTENT,
                kind=ArtifactDestinationKind.LOCAL_DIRECTORY,
                display_name=None,
                run_id=None,
            )
            self._persistent[existing.destination_id] = existing
            created = True
        else:
            self._verify_grant(existing)
        previous_default = self._default_id
        self._default_id = existing.destination_id
        try:
            cancelled = await self._persist_config()
        except BaseException:
            self._default_id = previous_default
            if created:
                self._persistent.pop(existing.destination_id, None)
            raise
        if cancelled:
            raise asyncio.CancelledError
        return self._view(existing, is_default=True)

    async def reset_export_destination(self) -> ArtifactDestination:
        self._require_config()
        previous_default = self._default_id
        self._default_id = None
        try:
            cancelled = await self._persist_config()
        except BaseException:
            self._default_id = previous_default
            raise
        if cancelled:
            raise asyncio.CancelledError
        return self._system_view()

    async def preflight_save(
        self,
        *,
        run_id: str,
        artifact_id: str,
        destination_id: str,
        filename: str | None,
    ) -> FrozenJsonObject:
        ref = await self.artifacts.find_ref(artifact_id)
        requested = self._filename_for_ref(ref, filename)
        try:
            grant = self._resolve(destination_id, run_id=run_id)
            self._verify_grant(grant)
        except ArtifactError as error:
            raise _retained_artifact_error(error, ref.artifact_id) from error
        return FrozenJsonObject.from_mapping(
            {
                "artifact_id": ref.artifact_id,
                "artifact_sha256": ref.sha256,
                "artifact_byte_size": ref.byte_size,
                "destination_id": grant.destination_id,
                "grant_digest": grant.grant_digest,
                "requested_filename": requested,
                "authorization": grant.authorization.value,
                "requires_approval": (
                    grant.authorization is DestinationAuthorization.ONE_TIME
                ),
            }
        )

    async def save_committed(
        self,
        *,
        run_id: str,
        artifact_id: str,
        destination_id: str,
        filename: str | None,
    ) -> ArtifactDeliveryReceipt:
        ref = await self.artifacts.find_ref(artifact_id)
        requested = self._filename_for_ref(ref, filename)
        try:
            grant = self._resolve(destination_id, run_id=run_id)
            self._verify_grant(grant)
        except ArtifactError as error:
            raise _retained_artifact_error(error, ref.artifact_id) from error
        payload = await self.artifacts.read_ref(ref)
        return await self._deliver(payload, grant, requested)

    async def save_public(
        self,
        artifact_id: str,
        *,
        destination: Path | None,
        filename: str | None,
    ) -> ArtifactDeliveryReceipt:
        ref = await self.artifacts.find_ref(artifact_id)
        requested = self._filename_for_ref(ref, filename)
        try:
            if destination is None:
                grant = self._resolve(DEFAULT_DESTINATION_SELECTOR, run_id=None)
            else:
                grant = await self._admit_path(
                    destination,
                    destination_id=self._next_destination_id(),
                    authorization=DestinationAuthorization.ONE_TIME,
                    kind=ArtifactDestinationKind.LOCAL_DIRECTORY,
                    display_name=None,
                    run_id=None,
                )
            self._verify_grant(grant)
        except ArtifactError as error:
            raise _retained_artifact_error(error, ref.artifact_id) from error
        payload = await self.artifacts.read_ref(ref)
        return await self._deliver(payload, grant, requested)

    async def preflight_set_default(self, destination_id: str) -> FrozenJsonObject:
        self._require_config()
        if destination_id == DEFAULT_DESTINATION_SELECTOR:
            raise ArtifactError(
                "artifact_destination_unauthorized",
                "The default selector cannot become a persistent destination.",
                {"destination_id": destination_id},
            )
        grant = self._resolve(destination_id, run_id=None, permit_one_time=False)
        if grant.authorization is DestinationAuthorization.ONE_TIME:
            raise ArtifactError(
                "artifact_destination_unauthorized",
                "A one-time destination cannot become the default.",
                {"destination_id": destination_id, "display_name": grant.display_name},
            )
        self._verify_grant(grant)
        return FrozenJsonObject.from_mapping(
            {
                "destination_id": grant.destination_id,
                "display_name": grant.display_name,
                "grant_digest": grant.grant_digest,
                "current_default_id": self._default_id,
                "delivery_config_sha256": self._config_digest(),
            }
        )

    async def set_default_by_id(self, destination_id: str) -> ArtifactDestination:
        self._require_config()
        grant = self._resolve(destination_id, run_id=None, permit_one_time=False)
        self._verify_grant(grant)
        previous_default = self._default_id
        if grant.authorization is DestinationAuthorization.SYSTEM:
            self._default_id = None
        elif grant.authorization is DestinationAuthorization.PERSISTENT:
            self._default_id = grant.destination_id
        else:
            raise ArtifactError(
                "artifact_destination_unauthorized",
                "A one-time destination cannot become the default.",
                {"destination_id": grant.destination_id},
            )
        try:
            cancelled = await self._persist_config()
        except BaseException:
            self._default_id = previous_default
            raise
        if cancelled:
            raise asyncio.CancelledError
        return self._view(grant, is_default=True)

    def approval_prompt_for_default(self, fingerprint: Mapping[str, object]) -> str:
        display_name = fingerprint.get("display_name")
        if not isinstance(display_name, str):
            raise ArtifactError(
                "artifact_destination_unauthorized",
                "The destination display name is unavailable.",
                {},
            )
        return f"Make “{display_name}” the default location for future exports?"

    def _resolve(
        self,
        destination_id: str,
        *,
        run_id: str | None,
        permit_one_time: bool = True,
    ) -> _DestinationGrant:
        self._require_config()
        selected = destination_id
        if destination_id == DEFAULT_DESTINATION_SELECTOR:
            selected = self._default_id or SYSTEM_DOWNLOADS_DESTINATION_ID
        if selected == SYSTEM_DOWNLOADS_DESTINATION_ID:
            if self._system is None:
                raise ArtifactError(
                    "artifact_downloads_unavailable",
                    "The operating system Downloads folder is unavailable.",
                    {},
                )
            return self._system
        grant = self._persistent.get(selected)
        if grant is not None:
            return grant
        grant = self._one_time.get(selected)
        if grant is not None and permit_one_time and grant.run_id == run_id:
            return grant
        raise ArtifactError(
            "artifact_destination_unauthorized",
            "The requested artifact destination is not authorized.",
            {"destination_id": destination_id},
        )

    async def _admit_path(
        self,
        directory: Path,
        *,
        destination_id: str,
        authorization: DestinationAuthorization,
        kind: ArtifactDestinationKind,
        display_name: str | None,
        run_id: str | None,
    ) -> _DestinationGrant:
        if not isinstance(directory, Path):
            raise ArtifactError(
                "artifact_destination_unauthorized",
                "Artifact destinations must be supplied as exact local directories.",
                {"destination_id": destination_id},
            )
        try:
            original = directory.absolute()
            original_facts = original.lstat()
            if original.is_symlink() or not stat.S_ISDIR(original_facts.st_mode):
                raise OSError("destination is not an exact directory")
            resolved = original.resolve(strict=True)
            facts = resolved.stat()
        except OSError as error:
            raise ArtifactError(
                (
                    "artifact_downloads_unavailable"
                    if authorization is DestinationAuthorization.SYSTEM
                    else "artifact_destination_unavailable"
                ),
                "The artifact destination is unavailable.",
                (
                    {}
                    if authorization is DestinationAuthorization.SYSTEM
                    else {"destination_id": destination_id}
                ),
            ) from error
        if resolved == self.agent_home or self.agent_home in resolved.parents:
            raise ArtifactError(
                "artifact_destination_unauthorized",
                "The artifact destination is not eligible.",
                {"destination_id": destination_id},
            )
        if not os.access(resolved, os.W_OK):
            raise ArtifactError(
                (
                    "artifact_downloads_unavailable"
                    if authorization is DestinationAuthorization.SYSTEM
                    else "artifact_destination_unavailable"
                ),
                "The artifact destination is not writable.",
                (
                    {}
                    if authorization is DestinationAuthorization.SYSTEM
                    else {"destination_id": destination_id}
                ),
            )
        authorized_at = self._clock()
        safe_display = display_name or _display_name(resolved)
        identity = {"device": int(facts.st_dev), "inode": int(facts.st_ino)}
        digest = (
            "sha256:"
            + sha256(
                canonical_json(
                    {
                        "destination_id": destination_id,
                        "path": str(resolved),
                        "identity": identity,
                        "authorization": authorization.value,
                        "run_id": run_id,
                    }
                ).encode("utf-8")
            ).hexdigest()
        )
        return _DestinationGrant(
            destination_id=destination_id,
            display_name=safe_display,
            kind=kind,
            authorization=authorization,
            path=resolved,
            device=identity["device"],
            inode=identity["inode"],
            grant_digest=digest,
            authorized_at=authorized_at,
            run_id=run_id,
        )

    def _verify_grant(self, grant: _DestinationGrant) -> None:
        try:
            facts = grant.path.lstat()
        except OSError as error:
            self._destination_failure(grant, unavailable=True, cause=error)
        if grant.path.is_symlink() or not stat.S_ISDIR(facts.st_mode):
            self._destination_failure(grant, unavailable=False)
        if int(facts.st_dev) != grant.device or int(facts.st_ino) != grant.inode:
            self._destination_failure(grant, unavailable=False)
        if not os.access(grant.path, os.W_OK):
            self._destination_failure(grant, unavailable=True)

    def _destination_failure(
        self,
        grant: _DestinationGrant,
        *,
        unavailable: bool,
        cause: BaseException | None = None,
    ) -> NoReturn:
        if grant.authorization is DestinationAuthorization.SYSTEM:
            error = ArtifactError(
                "artifact_downloads_unavailable",
                "The operating system Downloads folder is unavailable.",
                {},
            )
        else:
            error = ArtifactError(
                (
                    "artifact_destination_unavailable"
                    if unavailable
                    else "artifact_destination_revoked"
                ),
                (
                    "The artifact destination is unavailable."
                    if unavailable
                    else "The artifact destination authorization was revoked."
                ),
                {
                    "destination_id": grant.destination_id,
                    "display_name": grant.display_name,
                },
            )
        if cause is not None:
            error.__cause__ = cause
        raise error

    def _filename_for_ref(self, ref: ArtifactRef, filename: str | None) -> str:
        requested = ref.filename if filename is None else filename
        extension = "." + ref.filename.rsplit(".", 1)[1].casefold()
        return canonical_artifact_filename(
            requested,
            ref.media_type,
            ((ref.media_type, (extension,)),),
        )

    async def _deliver(
        self,
        payload: ArtifactPayload,
        grant: _DestinationGrant,
        filename: str,
    ) -> ArtifactDeliveryReceipt:
        gate = _DeliveryGate()
        worker = asyncio.create_task(
            asyncio.to_thread(self._deliver_sync, payload, grant, filename, gate)
        )
        cancelled = False
        try:
            async with asyncio.timeout(_DELIVERY_TIMEOUT_SECONDS):
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
                "artifact_delivery_failed",
                "Artifact delivery exceeded its I/O limit.",
                {
                    "artifact_id": payload.ref.artifact_id,
                    "destination_id": grant.destination_id,
                    "stage": "timeout",
                    "artifact_retained": True,
                },
            )
        try:
            receipt = worker.result()
        except _CancelledBeforePublication:
            raise asyncio.CancelledError from None
        except ArtifactError as error:
            raise _retained_artifact_error(error, payload.ref.artifact_id) from error
        if cancelled:
            raise asyncio.CancelledError
        return receipt

    def _deliver_sync(
        self,
        payload: ArtifactPayload,
        grant: _DestinationGrant,
        filename: str,
        gate: _DeliveryGate,
    ) -> ArtifactDeliveryReceipt:
        descriptor = -1
        temp_name: str | None = None
        final_name: str | None = None
        published = False
        try:
            flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(grant.path, flags)
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISDIR(opened.st_mode)
                or int(opened.st_dev) != grant.device
                or int(opened.st_ino) != grant.inode
            ):
                self._destination_failure(grant, unavailable=False)
            gate.require_active()
            temp_name = f".daita-artifact-{uuid4().hex}.tmp"
            temp_flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            temp = os.open(temp_name, temp_flags, 0o600, dir_fd=descriptor)
            try:
                digest = sha256()
                total = 0
                view = memoryview(payload.content)
                while view:
                    gate.require_active()
                    written = os.write(temp, view[: 1024 * 1024])
                    if written <= 0:
                        raise OSError("artifact delivery write made no progress")
                    digest.update(view[:written])
                    total += written
                    view = view[written:]
                os.fsync(temp)
                os.fchmod(temp, 0o600)
                if total != payload.ref.byte_size or (
                    "sha256:" + digest.hexdigest() != payload.ref.sha256
                ):
                    raise OSError("artifact delivery verification mismatch")
            finally:
                os.close(temp)
            assert temp_name is not None
            staged_name = temp_name
            stem, extension = filename.rsplit(".", 1)
            for suffix in range(MAX_COLLISION_SUFFIX + 1):
                candidate = (
                    filename if suffix == 0 else f"{stem} ({suffix}).{extension}"
                )
                try:

                    def publish() -> None:
                        nonlocal final_name, published
                        os.link(
                            staged_name,
                            candidate,
                            src_dir_fd=descriptor,
                            dst_dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                        final_name = candidate
                        published = True

                    gate.publish(publish)
                    break
                except FileExistsError:
                    continue
            if not published or final_name is None:
                raise ArtifactError(
                    "artifact_name_collision",
                    "No collision-free artifact filename is available.",
                    {
                        "artifact_id": payload.ref.artifact_id,
                        "requested_filename": filename,
                    },
                )
            os.unlink(temp_name, dir_fd=descriptor)
            temp_name = None
            _fsync_descriptor(descriptor)
            current = grant.path.lstat()
            if (
                grant.path.is_symlink()
                or int(current.st_dev) != grant.device
                or int(current.st_ino) != grant.inode
            ):
                os.unlink(final_name, dir_fd=descriptor)
                published = False
                self._destination_failure(grant, unavailable=False)
            final_flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
            final_flags |= getattr(os, "O_NOFOLLOW", 0)
            final = os.open(final_name, final_flags, dir_fd=descriptor)
            try:
                facts = os.fstat(final)
                if not stat.S_ISREG(facts.st_mode):
                    raise OSError("published artifact is not a regular file")
                digest = sha256()
                total = 0
                while True:
                    chunk = os.read(final, 1024 * 1024)
                    if not chunk:
                        break
                    digest.update(chunk)
                    total += len(chunk)
                if total != payload.ref.byte_size or (
                    "sha256:" + digest.hexdigest() != payload.ref.sha256
                ):
                    raise OSError("published artifact verification mismatch")
            finally:
                os.close(final)
            saved = (grant.path / final_name).resolve(strict=True)
            if saved.parent != grant.path:
                raise OSError("published artifact escaped its authorized directory")
            return ArtifactDeliveryReceipt(
                artifact_id=payload.ref.artifact_id,
                destination_id=grant.destination_id,
                filename=final_name,
                saved_path=str(saved),
                byte_size=payload.ref.byte_size,
                sha256=payload.ref.sha256,
                renamed_for_collision=final_name != filename,
                delivered_at=self._clock(),
            )
        except _CancelledBeforePublication:
            raise
        except ArtifactError:
            raise
        except Exception as error:
            failed_after_publication = published
            if published and final_name is not None and descriptor >= 0:
                try:
                    os.unlink(final_name, dir_fd=descriptor)
                    published = False
                except OSError:
                    pass
            raise ArtifactError(
                "artifact_delivery_failed",
                "The artifact could not be delivered.",
                {
                    "artifact_id": payload.ref.artifact_id,
                    "destination_id": grant.destination_id,
                    "stage": "verification" if failed_after_publication else "copy",
                    "artifact_retained": True,
                },
            ) from error
        finally:
            if temp_name is not None and descriptor >= 0:
                try:
                    os.unlink(temp_name, dir_fd=descriptor)
                except OSError:
                    pass
            if descriptor >= 0:
                os.close(descriptor)

    def _system_view(self) -> ArtifactDestination:
        if self._system is None:
            return ArtifactDestination(
                destination_id=SYSTEM_DOWNLOADS_DESTINATION_ID,
                display_name="Downloads",
                kind=ArtifactDestinationKind.SYSTEM_DOWNLOADS,
                authorization=DestinationAuthorization.SYSTEM,
                availability=DestinationAvailability.UNAVAILABLE,
                is_default=self._default_id is None,
            )
        return self._view(self._system, is_default=self._default_id is None)

    def _view(
        self, grant: _DestinationGrant, *, is_default: bool
    ) -> ArtifactDestination:
        try:
            self._verify_grant(grant)
        except ArtifactError as error:
            availability = (
                DestinationAvailability.REVOKED
                if error.code == "artifact_destination_revoked"
                else DestinationAvailability.UNAVAILABLE
            )
        else:
            availability = DestinationAvailability.AVAILABLE
        return ArtifactDestination(
            destination_id=grant.destination_id,
            display_name=grant.display_name,
            kind=grant.kind,
            authorization=grant.authorization,
            availability=availability,
            is_default=is_default,
        )

    def _next_destination_id(self) -> str:
        destination_id = self._id_factory("destination")
        if (
            not isinstance(destination_id, str)
            or _DESTINATION_ID.fullmatch(destination_id) is None
        ):
            raise ArtifactError(
                "artifact_storage_failed",
                "The destination identity factory returned an invalid identity.",
                {"stage": "destination_identity"},
            )
        return destination_id

    def _require_config(self) -> None:
        if self._config_error is not None:
            raise ArtifactError(
                self._config_error.code,
                self._config_error.message,
                self._config_error.details,
            )

    def _load_config(self) -> None:
        if not self._config_path.exists():
            return
        try:
            facts = self._config_path.lstat()
            if not stat.S_ISREG(facts.st_mode) or self._config_path.is_symlink():
                raise ValueError("delivery config is not a regular file")
            if os.name != "nt" and stat.S_IMODE(facts.st_mode) != 0o600:
                raise ValueError("delivery config permissions are invalid")
            content = self._config_path.read_bytes()
            if len(content) > _MAX_CONFIG_BYTES:
                raise ValueError("delivery config exceeds its bound")
            raw = json.loads(content.decode("utf-8"))
            if not isinstance(raw, dict) or set(raw) != {
                "default_destination_id",
                "persistent_destinations",
            }:
                raise ValueError("delivery config shape is invalid")
            entries = raw["persistent_destinations"]
            if (
                not isinstance(entries, list)
                or len(entries) > MAX_PERSISTENT_DESTINATIONS
            ):
                raise ValueError("delivery destination count is invalid")
            persistent: dict[str, _DestinationGrant] = {}
            for item in entries:
                if not isinstance(item, dict):
                    raise ValueError("delivery destination entry is invalid")
                grant = _grant_from_mapping(item)
                if grant.destination_id in persistent:
                    raise ValueError("delivery destination identity duplicates")
                persistent[grant.destination_id] = grant
            default_id = raw["default_destination_id"]
            if default_id is not None and default_id not in persistent:
                raise ValueError("delivery default is not a persistent destination")
            if content != canonical_json(raw).encode("utf-8"):
                raise ValueError("delivery config is not canonical")
            self._persistent = persistent
            self._default_id = default_id
        except Exception as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "The export-location configuration is invalid.",
                {"stage": "delivery_config"},
            ) from error

    async def _persist_config(self) -> bool:
        """Finish the bounded atomic write and report any late cancellation."""

        worker = asyncio.create_task(asyncio.to_thread(self._write_config))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        worker.result()
        return cancelled

    def _write_config(self) -> None:
        try:
            self._write_config_unchecked()
        except ArtifactError:
            raise
        except Exception as error:
            raise ArtifactError(
                "artifact_storage_failed",
                "The export-location configuration could not be saved.",
                {"stage": "delivery_config"},
            ) from error

    def _write_config_unchecked(self) -> None:
        content = canonical_json(self._config_mapping()).encode("utf-8")
        temporary = self.artifacts.staging / f"delivery-config.{uuid4().hex}.tmp"
        try:
            flags = (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            descriptor = os.open(temporary, flags, 0o600)
            try:
                view = memoryview(content)
                while view:
                    written = os.write(descriptor, view)
                    if written <= 0:
                        raise OSError("delivery config write made no progress")
                    view = view[written:]
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o600)
            finally:
                os.close(descriptor)
            os.replace(temporary, self._config_path)
            _fsync_directory(self._config_path.parent)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def _config_mapping(self) -> dict[str, object]:
        return {
            "default_destination_id": self._default_id,
            "persistent_destinations": tuple(
                _grant_to_mapping(item)
                for item in sorted(
                    self._persistent.values(), key=lambda grant: grant.destination_id
                )
            ),
        }

    def _config_digest(self) -> str:
        return (
            "sha256:"
            + sha256(canonical_json(self._config_mapping()).encode("utf-8")).hexdigest()
        )


def _grant_to_mapping(grant: _DestinationGrant) -> dict[str, object]:
    return {
        "destination_id": grant.destination_id,
        "display_name": grant.display_name,
        "kind": grant.kind.value,
        "authorization": grant.authorization.value,
        "path": str(grant.path),
        "device": grant.device,
        "inode": grant.inode,
        "grant_digest": grant.grant_digest,
        "authorized_at": grant.authorized_at.isoformat().replace("+00:00", "Z"),
    }


def _grant_from_mapping(value: Mapping[str, object]) -> _DestinationGrant:
    if set(value) != {
        "destination_id",
        "display_name",
        "kind",
        "authorization",
        "path",
        "device",
        "inode",
        "grant_digest",
        "authorized_at",
    }:
        raise ValueError("persistent destination grant shape is invalid")
    destination_id = value.get("destination_id")
    display_name = value.get("display_name")
    raw_kind = value.get("kind")
    raw_authorization = value.get("authorization")
    raw_path = value.get("path")
    grant_digest = value.get("grant_digest")
    authorized_at = value.get("authorized_at")
    if not all(
        isinstance(item, str)
        for item in (
            destination_id,
            display_name,
            raw_kind,
            raw_authorization,
            raw_path,
            grant_digest,
            authorized_at,
        )
    ):
        raise ValueError("persistent destination grant values are invalid")
    assert isinstance(destination_id, str)
    assert isinstance(display_name, str)
    assert isinstance(raw_kind, str)
    assert isinstance(raw_authorization, str)
    assert isinstance(raw_path, str)
    assert isinstance(grant_digest, str)
    assert isinstance(authorized_at, str)
    if not authorized_at.endswith("Z"):
        raise ValueError("destination authorized_at is invalid")
    moment = datetime.fromisoformat(authorized_at[:-1] + "+00:00")
    offset = moment.utcoffset()
    if (
        offset is None
        or offset.total_seconds() != 0
        or moment.isoformat().replace("+00:00", "Z") != authorized_at
    ):
        raise ValueError("destination authorized_at is not UTC")
    raw_device = value.get("device")
    raw_inode = value.get("inode")
    if (
        type(raw_device) is not int
        or type(raw_inode) is not int
        or raw_device < 0
        or raw_inode < 0
    ):
        raise ValueError("persistent destination identity is invalid")
    path = Path(raw_path)
    grant = _DestinationGrant(
        destination_id=destination_id,
        display_name=display_name,
        kind=ArtifactDestinationKind(raw_kind),
        authorization=DestinationAuthorization(raw_authorization),
        path=path,
        device=raw_device,
        inode=raw_inode,
        grant_digest=grant_digest,
        authorized_at=moment,
    )
    expected_digest = (
        "sha256:"
        + sha256(
            canonical_json(
                {
                    "destination_id": grant.destination_id,
                    "path": str(grant.path),
                    "identity": {"device": grant.device, "inode": grant.inode},
                    "authorization": grant.authorization.value,
                    "run_id": None,
                }
            ).encode("utf-8")
        ).hexdigest()
    )
    if (
        _DESTINATION_ID.fullmatch(grant.destination_id) is None
        or grant.authorization is not DestinationAuthorization.PERSISTENT
        or grant.kind is not ArtifactDestinationKind.LOCAL_DIRECTORY
        or not grant.path.is_absolute()
        or os.path.normpath(raw_path) != raw_path
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", grant.grant_digest)
        or grant.grant_digest != expected_digest
    ):
        raise ValueError("persistent destination grant is invalid")
    ArtifactDestination(
        destination_id=grant.destination_id,
        display_name=grant.display_name,
        kind=grant.kind,
        authorization=grant.authorization,
        availability=DestinationAvailability.AVAILABLE,
        is_default=False,
    )
    return grant


def _retained_artifact_error(error: ArtifactError, artifact_id: str) -> ArtifactError:
    details = error.details.to_dict()
    details.setdefault("artifact_id", artifact_id)
    details.setdefault("artifact_retained", True)
    return ArtifactError(error.code, error.message, details)


def _display_name(path: Path) -> str:
    candidate = path.name or path.anchor or "Authorized folder"
    projected = "".join(
        character
        for character in candidate
        if character.isprintable() and not _unsafe_display_character(character)
    )[:80]
    return projected or "Authorized folder"


def _unsafe_display_character(value: str) -> bool:
    import unicodedata

    return unicodedata.category(value) in {"Cc", "Cf", "Cs"}


def _fsync_descriptor(descriptor: int) -> None:
    if os.name != "nt":
        os.fsync(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt" or not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def resolve_os_downloads_directory() -> Path:
    """Resolve the platform Downloads known folder without a shell."""

    if sys.platform == "win32":
        return _windows_downloads()
    if sys.platform == "darwin":
        return _macos_downloads()
    return _xdg_downloads()


def _windows_downloads() -> Path:
    class _GUID(ctypes.Structure):
        _fields_ = [
            ("Data1", ctypes.c_ulong),
            ("Data2", ctypes.c_ushort),
            ("Data3", ctypes.c_ushort),
            ("Data4", ctypes.c_ubyte * 8),
        ]

    folder = _GUID(
        0x374DE290,
        0x123F,
        0x4565,
        (ctypes.c_ubyte * 8)(0x91, 0x64, 0x39, 0xC4, 0x92, 0x5E, 0x46, 0x7B),
    )
    result = ctypes.c_wchar_p()
    windows_libraries = getattr(ctypes, "windll")
    shell32 = windows_libraries.shell32
    ole32 = windows_libraries.ole32
    status = shell32.SHGetKnownFolderPath(
        ctypes.byref(folder), 0, None, ctypes.byref(result)
    )
    if status != 0 or not result.value:
        raise OSError("Downloads known folder is unavailable")
    try:
        return Path(result.value)
    finally:
        ole32.CoTaskMemFree(result)


def _macos_downloads() -> Path:
    foundation = ctypes.cdll.LoadLibrary(
        "/System/Library/Frameworks/Foundation.framework/Foundation"
    )
    objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
    foundation.NSSearchPathForDirectoriesInDomains.argtypes = (
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_bool,
    )
    foundation.NSSearchPathForDirectoriesInDomains.restype = ctypes.c_void_p
    array = foundation.NSSearchPathForDirectoriesInDomains(15, 1, True)
    if not array:
        raise OSError("Downloads search path is unavailable")
    objc.sel_registerName.argtypes = (ctypes.c_char_p,)
    objc.sel_registerName.restype = ctypes.c_void_p
    message_address = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
    if message_address is None:
        raise OSError("Objective-C runtime is unavailable")
    object_at_index = ctypes.CFUNCTYPE(
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong
    )(message_address)
    utf8_string = ctypes.CFUNCTYPE(ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p)(
        message_address
    )
    string = object_at_index(array, objc.sel_registerName(b"objectAtIndex:"), 0)
    encoded = utf8_string(string, objc.sel_registerName(b"UTF8String"))
    if not encoded:
        raise OSError("Downloads search path is unavailable")
    return Path(encoded.decode("utf-8"))


def _xdg_downloads() -> Path:
    home = Path.home()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", str(home / ".config")))
    document = config_home / "user-dirs.dirs"
    try:
        lines = document.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise OSError("XDG Downloads configuration is unavailable") from error
    prefix = "XDG_DOWNLOAD_DIR="
    for line in lines:
        if not line.startswith(prefix):
            continue
        raw = line[len(prefix) :].strip()
        if len(raw) < 2 or raw[0] != '"' or raw[-1] != '"':
            break
        value = raw[1:-1]
        value = value.replace("$HOME", str(home), 1)
        candidate = Path(value)
        if candidate.is_absolute():
            return candidate
        break
    raise OSError("XDG Downloads configuration is invalid")


__all__ = [
    "DeliverySourceReader",
    "LocalArtifactDelivery",
    "resolve_os_downloads_directory",
]
