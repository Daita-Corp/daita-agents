"""Sandboxed local-directory discovery and inspection adapter."""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
import stat
import threading
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path, PurePosixPath
from urllib.parse import quote

from .._json import canonical_json
from ..capabilities import ExtensionDeclarations
from ..catalog.models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSync,
    CatalogSyncStatus,
    FacetKind,
    FileFacet,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from ..catalog.protocols import CatalogStore
from ..domains.data.export_capabilities import (
    LocalFileCopyIncompleteError,
    LocalFileCopyResult,
)
from ..domains.data.file_capabilities import LocalFileReadResult
from ..domains.data.results import project_result_rows
from .models import (
    DiscoveryRequest,
    DiscoveryResult,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from .protocols import (
    DiscoveryLimitError,
    ResourceAdapterError,
    ResourceNotFoundError,
    SourceClosedError,
    SourceStore,
    StaleResourceError,
)

_ADAPTER_ID = "local-directory"
_SUPPORTED_SUFFIXES = frozenset({".csv", ".json"})
_READ_CHUNK_BYTES = 64 * 1024


class LocalDirectorySourceError(ResourceAdapterError):
    """Normalized local-directory failure without leaking file contents."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        source_id: str = "local-directory:unopened",
    ) -> None:
        super().__init__(source_id, code, message)


@dataclass(frozen=True, slots=True)
class LocalDirectorySource:
    """One explicitly attached canonical directory root."""

    root: str | Path
    name: str | None = None
    max_depth: int = 8
    max_files: int = 1_000
    max_file_bytes: int = 2 * 1024 * 1024
    max_columns: int = 512
    max_rows: int = 100_000
    max_json_nodes: int = 500_000
    max_json_depth: int = 32
    max_key_bytes: int = 1_024
    max_string_bytes: int = 256 * 1024
    max_cell_bytes: int = 1024 * 1024

    def __post_init__(self) -> None:
        if self.name is not None and (
            not isinstance(self.name, str)
            or not self.name.strip()
            or self.name != self.name.strip()
        ):
            raise ValueError("local-directory name must be non-empty and trimmed")
        _bounded_limit(self.max_depth, "max_depth", maximum=64)
        _bounded_limit(self.max_files, "max_files", maximum=10_000)
        _bounded_limit(
            self.max_file_bytes,
            "max_file_bytes",
            maximum=2 * 1024 * 1024,
        )
        _bounded_limit(self.max_columns, "max_columns", maximum=4_096)
        _bounded_limit(self.max_rows, "max_rows", maximum=1_000_000)
        _bounded_limit(self.max_json_nodes, "max_json_nodes", maximum=2_000_000)
        _bounded_limit(self.max_json_depth, "max_json_depth", maximum=128)
        _bounded_limit(self.max_key_bytes, "max_key_bytes", maximum=64 * 1024)
        _bounded_limit(
            self.max_string_bytes,
            "max_string_bytes",
            maximum=8 * 1024 * 1024,
        )
        _bounded_limit(
            self.max_cell_bytes,
            "max_cell_bytes",
            maximum=8 * 1024 * 1024,
        )

    async def open(
        self,
        *,
        agent_id: str,
        attached_at: datetime,
        clock,
    ) -> LocalDirectoryResourceAdapter:
        worker = asyncio.create_task(asyncio.to_thread(_open_root, self.root))
        try:
            root = await asyncio.shield(worker)
        except asyncio.CancelledError:
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            if not worker.cancelled() and worker.exception() is None:
                os.close(worker.result().descriptor)
            raise

        display_name = self.name or root.path.name
        try:
            registration = SourceRegistration.build(
                agent_id=agent_id,
                adapter_id=_ADAPTER_ID,
                native_identity=str(root.path),
                display_name=display_name,
                configuration={
                    "root": str(root.path),
                    "root_device": root.device,
                    "root_inode": root.inode,
                    "max_depth": self.max_depth,
                    "max_files": self.max_files,
                    "max_file_bytes": self.max_file_bytes,
                    "max_columns": self.max_columns,
                    "max_rows": self.max_rows,
                    "max_json_nodes": self.max_json_nodes,
                    "max_json_depth": self.max_json_depth,
                    "max_key_bytes": self.max_key_bytes,
                    "max_string_bytes": self.max_string_bytes,
                    "max_cell_bytes": self.max_cell_bytes,
                    "formats": ("csv", "json"),
                },
                attached_at=attached_at,
            )
        except BaseException:
            os.close(root.descriptor)
            raise
        return LocalDirectoryResourceAdapter(
            registration=registration,
            root=root,
            clock=clock,
            limits=_DiscoveryLimits(
                max_depth=self.max_depth,
                max_files=self.max_files,
                max_file_bytes=self.max_file_bytes,
                max_columns=self.max_columns,
                max_rows=self.max_rows,
                max_json_nodes=self.max_json_nodes,
                max_json_depth=self.max_json_depth,
                max_key_bytes=self.max_key_bytes,
                max_string_bytes=self.max_string_bytes,
                max_cell_bytes=self.max_cell_bytes,
            ),
        )


class LocalDirectoryResourceAdapter:
    """Bounded control-plane view over one already-admitted directory root."""

    def __init__(
        self,
        *,
        registration: SourceRegistration,
        root: _RootDescriptor,
        clock,
        limits: _DiscoveryLimits,
    ) -> None:
        self._registration = registration
        self._root = root
        self._clock = clock
        self._limits = limits
        self._lock = asyncio.Lock()
        self._closed = False
        self._latest: SourceCatalogSnapshot | None = None

    @property
    def registration(self) -> SourceRegistration:
        return self._registration

    def declarations(self) -> ExtensionDeclarations:
        from ..domains.data.file_capabilities import (
            local_file_read_extension_declarations,
        )

        return local_file_read_extension_declarations()

    async def discover(self, request: DiscoveryRequest) -> DiscoveryResult:
        self._require_request(request)
        async with self._lock:
            self._require_open()
            cancellation = threading.Event()
            descriptor = os.dup(self._root.descriptor)
            worker = asyncio.create_task(
                asyncio.to_thread(
                    _discover,
                    descriptor,
                    self._root,
                    self._registration,
                    request,
                    self._limits,
                    self._clock(),
                    cancellation,
                )
            )
            try:
                snapshot = await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancellation.set()
                while not worker.done():
                    try:
                        await asyncio.shield(worker)
                    except asyncio.CancelledError:
                        continue
                    except BaseException:
                        break
                raise
            self._latest = snapshot
            assert snapshot.sync.completed_at is not None
            return DiscoveryResult(
                request=request,
                snapshot=snapshot,
                completed_at=snapshot.sync.completed_at,
            )

    async def inspect(self, resource: ResourceRef) -> ResourceSnapshot:
        if not isinstance(resource, ResourceRef):
            raise TypeError("resource must be a ResourceRef")
        async with self._lock:
            self._require_open()
            snapshot = self._latest
            if (
                resource.agent_id != self._registration.agent_id
                or resource.source_id != self._registration.id
                or snapshot is None
            ):
                raise ResourceNotFoundError(
                    self._registration.id,
                    resource.resource_id,
                )
            current = next(
                (
                    item
                    for item in snapshot.resources
                    if item.id == resource.resource_id
                ),
                None,
            )
            if current is None:
                raise ResourceNotFoundError(
                    self._registration.id,
                    resource.resource_id,
                )
            if resource.revision is not None and (
                resource.revision != current.current_revision
            ):
                raise StaleResourceError(
                    self._registration.id,
                    resource.resource_id,
                )
            revision = next(
                item for item in snapshot.revisions if item.resource_id == current.id
            )
            return ResourceSnapshot(
                reference=resource,
                resource=current,
                revision=revision,
                facets=tuple(
                    facet
                    for facet in snapshot.facets
                    if facet.resource_id == current.id
                ),
                relationships=tuple(
                    relationship
                    for relationship in snapshot.relationships
                    if current.id
                    in {
                        relationship.from_resource_id,
                        relationship.to_resource_id,
                    }
                ),
                inspected_at=self._clock(),
                source_revision=revision.source_revision,
            )

    async def health(self) -> SourceHealth:
        async with self._lock:
            checked_at = self._clock()
            if self._closed:
                return SourceHealth(
                    agent_id=self._registration.agent_id,
                    source_id=self._registration.id,
                    adapter_id=_ADAPTER_ID,
                    healthy=False,
                    checked_at=checked_at,
                    error_code="source_closed",
                )
            try:
                await asyncio.to_thread(_verify_root_path, self._root)
            except LocalDirectorySourceError:
                return SourceHealth(
                    agent_id=self._registration.agent_id,
                    source_id=self._registration.id,
                    adapter_id=_ADAPTER_ID,
                    healthy=False,
                    checked_at=checked_at,
                    error_code="local_root_unavailable",
                )
            return SourceHealth(
                agent_id=self._registration.agent_id,
                source_id=self._registration.id,
                adapter_id=_ADAPTER_ID,
                healthy=True,
                checked_at=checked_at,
                source_revision=_root_revision(self._root),
                details={"canonical_root": True},
            )

    async def close(self) -> None:
        async with self._lock:
            if self._closed:
                return
            os.close(self._root.descriptor)
            self._closed = True

    def _require_request(self, request: DiscoveryRequest) -> None:
        if not isinstance(request, DiscoveryRequest):
            raise TypeError("request must be a DiscoveryRequest")
        if (
            request.agent_id != self._registration.agent_id
            or request.source_id != self._registration.id
        ):
            raise ResourceAdapterError(
                self._registration.id,
                "source_scope_mismatch",
                "discovery request does not match this source",
            )

    def _require_open(self) -> None:
        if self._closed:
            raise SourceClosedError(self._registration.id)


class LocalDirectoryReadBackend:
    """Revalidate persisted file scope, then perform one bounded safe read."""

    def __init__(self, sources: SourceStore, catalog: CatalogStore) -> None:
        if not isinstance(sources, SourceStore):
            raise TypeError("sources must implement SourceStore")
        if not isinstance(catalog, CatalogStore):
            raise TypeError("catalog must implement CatalogStore")
        self._sources = sources
        self._catalog = catalog

    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        max_rows: int,
        max_bytes: int,
    ) -> LocalFileReadResult:
        _required_identifier(agent_id, "agent_id")
        _required_identifier(source_id, "source_id")
        _required_identifier(resource_id, "resource_id")
        _bounded_limit(max_rows, "max_rows", maximum=1_000_000)
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or not 2 <= max_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("max_bytes must be from 2 through 67108864")

        registration = await self._sources.load_source(agent_id, source_id)
        if registration is None or not registration.active:
            raise LocalDirectorySourceError(
                "source_not_available",
                "The local-directory source is not attached to this agent.",
                source_id=source_id,
            )
        if registration.adapter_id != _ADAPTER_ID:
            raise LocalDirectorySourceError(
                "source_adapter_mismatch",
                "The selected source is not a local-directory source.",
                source_id=source_id,
            )
        resource = await self._catalog.load_resource(agent_id, resource_id)
        if resource is None:
            raise LocalDirectorySourceError(
                "resource_not_available",
                "The selected file is not present in the current catalog.",
                source_id=source_id,
            )
        if resource.source_id != source_id or resource.agent_id != agent_id:
            raise LocalDirectorySourceError(
                "resource_scope_mismatch",
                "The selected file does not belong to this source.",
                source_id=source_id,
            )
        if resource.kind is not ResourceKind.FILE:
            raise LocalDirectorySourceError(
                "resource_kind_unsupported",
                "Local reads require a current CSV or JSON file resource.",
                source_id=source_id,
            )
        relative_path = _normalized_relative_path(resource.native_identity, source_id)
        revision = await self._catalog.load_revision(
            agent_id,
            resource.id,
            resource.current_revision,
        )
        if (
            revision is None
            or revision.sync_id != resource.current_sync_id
            or revision.revision != resource.current_revision
        ):
            raise LocalDirectorySourceError(
                "catalog_revision_missing",
                "The selected file has no current catalog revision.",
                source_id=source_id,
            )
        facets = await self._catalog.load_facets(
            agent_id,
            resource.id,
            resource.current_revision,
        )
        if {item.revision for item in facets} != set(revision.facet_revisions) or any(
            item.resource_id != resource.id or item.sync_id != resource.current_sync_id
            for item in facets
        ):
            raise LocalDirectorySourceError(
                "catalog_facets_invalid",
                "The current file catalog facts are incomplete or inconsistent.",
                source_id=source_id,
            )
        file_facet, tabular_facet = _current_file_facets(facets, source_id)
        limits, root_path, root_device, root_inode = _read_configuration(registration)
        expected = _expected_file_facts(
            file_facet,
            tabular_facet,
            revision.source_revision,
            source_id,
        )
        if PurePosixPath(relative_path).suffix.casefold() != f".{expected.format}":
            raise LocalDirectorySourceError(
                "catalog_file_format_invalid",
                "The catalog file format does not match its resource identity.",
                source_id=source_id,
            )

        cancellation = threading.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(
                _read_registered_file,
                root_path,
                root_device,
                root_inode,
                relative_path,
                limits.max_file_bytes,
                source_id,
                cancellation,
            )
        )
        try:
            content, _file_stat = await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancellation.set()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            raise
        except LocalDirectorySourceError:
            raise
        except (OSError, RuntimeError) as error:
            raise LocalDirectorySourceError(
                "local_file_read_failed",
                "The local file could not be read safely.",
                source_id=source_id,
            ) from error

        content_hash = "sha256:" + sha256(content).hexdigest()
        if (
            len(content) != expected.size_bytes
            or content_hash != expected.content_sha256
            or revision.source_revision != expected.content_sha256
        ):
            raise LocalDirectorySourceError(
                "catalog_file_stale",
                "The local file changed after its catalog snapshot.",
                source_id=source_id,
            )
        try:
            parsed = _parse_tabular(
                content,
                expected.format,
                limits,
            )
        except (
            UnicodeError,
            csv.Error,
            json.JSONDecodeError,
            RecursionError,
            ValueError,
        ) as error:
            raise LocalDirectorySourceError(
                "local_file_invalid",
                "The local file violates its bounded CSV/JSON contract.",
                source_id=source_id,
            ) from error
        if parsed.encoding != expected.encoding:
            raise LocalDirectorySourceError(
                "catalog_file_stale",
                "The local file encoding changed after catalog discovery.",
                source_id=source_id,
            )
        if parsed.columns != expected.columns:
            raise LocalDirectorySourceError(
                "catalog_file_stale",
                "The local file schema changed after catalog discovery.",
                source_id=source_id,
            )
        projection = project_result_rows(
            parsed.rows,
            max_rows=max_rows,
            max_bytes=max_bytes,
        )
        return LocalFileReadResult(
            source_id=source_id,
            source_revision=expected.content_sha256,
            resource_id=resource.id,
            resource_revision=resource.current_revision,
            format=parsed.format,
            encoding=parsed.encoding,
            columns=parsed.columns,
            projection=projection,
        )

    async def execute_copy(
        self,
        *,
        agent_id: str,
        source_id: str,
        resource_id: str,
        max_bytes: int,
    ) -> LocalFileCopyResult:
        """Read exact current source bytes without parsing or converting them."""

        _required_identifier(agent_id, "agent_id")
        _required_identifier(source_id, "source_id")
        _required_identifier(resource_id, "resource_id")
        if (
            not isinstance(max_bytes, int)
            or isinstance(max_bytes, bool)
            or not 1 <= max_bytes <= 64 * 1024 * 1024
        ):
            raise ValueError("max_bytes must be from 1 through 67108864")

        registration = await self._sources.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.agent_id != agent_id
            or registration.id != source_id
            or not registration.active
        ):
            raise LocalDirectorySourceError(
                "source_not_available",
                "The local-directory source is not attached to this agent.",
                source_id=source_id,
            )
        if registration.adapter_id != _ADAPTER_ID:
            raise LocalDirectorySourceError(
                "source_adapter_mismatch",
                "The selected source is not a local-directory source.",
                source_id=source_id,
            )
        resource = await self._catalog.load_resource(agent_id, resource_id)
        if resource is None or resource.id != resource_id:
            raise LocalDirectorySourceError(
                "resource_not_available",
                "The selected file is not present in the current catalog.",
                source_id=source_id,
            )
        if resource.source_id != source_id or resource.agent_id != agent_id:
            raise LocalDirectorySourceError(
                "resource_scope_mismatch",
                "The selected file does not belong to this source.",
                source_id=source_id,
            )
        if resource.kind is not ResourceKind.FILE:
            raise LocalDirectorySourceError(
                "resource_kind_unsupported",
                "Local copies require a current CSV or JSON file resource.",
                source_id=source_id,
            )
        relative_path = _normalized_relative_path(resource.native_identity, source_id)
        revision = await self._catalog.load_revision(
            agent_id,
            resource.id,
            resource.current_revision,
        )
        if (
            revision is None
            or revision.resource_id != resource.id
            or revision.sync_id != resource.current_sync_id
            or revision.revision != resource.current_revision
        ):
            raise LocalDirectorySourceError(
                "catalog_revision_missing",
                "The selected file has no current catalog revision.",
                source_id=source_id,
            )
        sync = await self._catalog.load_sync(agent_id, resource.current_sync_id)
        if (
            sync is None
            or sync.id != resource.current_sync_id
            or sync.agent_id != agent_id
            or sync.source_id != source_id
            or sync.status is not CatalogSyncStatus.SUCCEEDED
            or sync.source_revision is None
        ):
            raise LocalDirectorySourceError(
                "catalog_revision_missing",
                "The selected file has no current source revision.",
                source_id=source_id,
            )
        facets = await self._catalog.load_facets(
            agent_id,
            resource.id,
            resource.current_revision,
        )
        if {item.revision for item in facets} != set(revision.facet_revisions) or any(
            item.resource_id != resource.id or item.sync_id != resource.current_sync_id
            for item in facets
        ):
            raise LocalDirectorySourceError(
                "catalog_facets_invalid",
                "The current file catalog facts are incomplete or inconsistent.",
                source_id=source_id,
            )
        file_facet, tabular_facet = _current_file_facets(facets, source_id)
        limits, root_path, root_device, root_inode = _read_configuration(registration)
        expected = _expected_file_facts(
            file_facet,
            tabular_facet,
            revision.source_revision,
            source_id,
        )
        if PurePosixPath(relative_path).suffix.casefold() != f".{expected.format}":
            raise LocalDirectorySourceError(
                "catalog_file_format_invalid",
                "The catalog file format does not match its resource identity.",
                source_id=source_id,
            )
        effective_max_bytes = min(max_bytes, limits.max_file_bytes)
        if expected.size_bytes > effective_max_bytes:
            raise LocalFileCopyIncompleteError("byte_limit")

        cancellation = threading.Event()
        worker = asyncio.create_task(
            asyncio.to_thread(
                _read_registered_file,
                root_path,
                root_device,
                root_inode,
                relative_path,
                effective_max_bytes,
                source_id,
                cancellation,
            )
        )
        try:
            content, _file_stat = await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancellation.set()
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
            raise
        except DiscoveryLimitError as error:
            raise LocalFileCopyIncompleteError("byte_limit") from error
        except LocalDirectorySourceError:
            raise
        except (OSError, RuntimeError) as error:
            raise LocalDirectorySourceError(
                "local_file_read_failed",
                "The local file could not be copied safely.",
                source_id=source_id,
            ) from error

        content_hash = "sha256:" + sha256(content).hexdigest()
        if (
            len(content) != expected.size_bytes
            or content_hash != expected.content_sha256
            or revision.source_revision != expected.content_sha256
        ):
            raise LocalFileCopyIncompleteError(
                "source_changed",
                completed_bytes=len(content),
            )
        return LocalFileCopyResult(
            source_id=source_id,
            source_revision=sync.source_revision,
            resource_id=resource.id,
            resource_revision=resource.current_revision,
            filename=PurePosixPath(relative_path).name,
            format=expected.format,
            media_type=expected.media_type,
            content=content,
            sensitivity=resource.sensitivity,
        )


@dataclass(frozen=True, slots=True)
class _RootDescriptor:
    path: Path
    descriptor: int
    device: int
    inode: int


@dataclass(frozen=True, slots=True)
class _DiscoveryLimits:
    max_depth: int
    max_files: int
    max_file_bytes: int
    max_columns: int
    max_rows: int
    max_json_nodes: int
    max_json_depth: int
    max_key_bytes: int
    max_string_bytes: int
    max_cell_bytes: int


@dataclass(frozen=True, slots=True)
class _FileFacts:
    relative_path: str
    content_sha256: str
    file_facet: CatalogFacet
    tabular_facet: CatalogFacet


@dataclass(frozen=True, slots=True)
class _ParsedFile:
    format: str
    encoding: str
    columns: tuple[str, ...]
    rows: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class _ExpectedFileFacts:
    format: str
    media_type: str
    encoding: str
    size_bytes: int
    content_sha256: str
    columns: tuple[str, ...]


def _bounded_limit(value: int, name: str, *, maximum: int) -> None:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or not 1 <= value <= maximum
    ):
        raise ValueError(f"{name} must be from 1 through {maximum}")


def _required_identifier(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or value != value.strip()
        or len(value) > 512
    ):
        raise ValueError(f"{name} must be a bounded non-empty string")


def _root_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _open_root(value: str | Path) -> _RootDescriptor:
    raw = os.fspath(value)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise LocalDirectorySourceError(
            "local_root_invalid",
            "Local-directory root must be an existing canonical directory.",
        )
    candidate = Path(raw)
    if ".." in candidate.parts:
        raise LocalDirectorySourceError(
            "local_root_invalid",
            "Local-directory root must be an existing canonical directory.",
        )
    lexical = Path(os.path.abspath(raw))
    if not lexical.is_absolute() or lexical.anchor != os.path.sep:
        raise LocalDirectorySourceError(
            "local_root_invalid",
            "Local-directory root must be an existing canonical directory.",
        )

    descriptors: list[int] = []
    bindings: list[tuple[int, str, int]] = []
    try:
        anchor = lexical.anchor
        descriptor = os.open(anchor, _root_flags())
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        direct = os.stat(anchor, follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or not _same_identity(opened, direct):
            raise OSError("root anchor is not a stable directory")

        for component in lexical.parts[1:]:
            direct = os.stat(
                component,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
            if not stat.S_ISDIR(direct.st_mode):
                raise OSError("root component is not a directory")
            child = os.open(component, _root_flags(), dir_fd=descriptor)
            descriptors.append(child)
            child_opened = os.fstat(child)
            if not stat.S_ISDIR(child_opened.st_mode) or not _same_identity(
                direct,
                child_opened,
            ):
                raise OSError("root component changed during admission")
            bindings.append((descriptor, component, child))
            descriptor = child
            opened = child_opened

        # Keep every ancestor descriptor open until all pathname bindings have
        # been rechecked. This prevents a component swapped after its own open
        # from silently changing which directory the explicit root names.
        for parent, component, child in bindings:
            current = os.stat(
                component,
                dir_fd=parent,
                follow_symlinks=False,
            )
            child_opened = os.fstat(child)
            if not stat.S_ISDIR(current.st_mode) or not _same_identity(
                current,
                child_opened,
            ):
                raise OSError("root component changed during admission")

        for ancestor in descriptors[:-1]:
            os.close(ancestor)
        return _RootDescriptor(
            path=lexical,
            descriptor=descriptor,
            device=int(opened.st_dev),
            inode=int(opened.st_ino),
        )
    except OSError as error:
        for descriptor in reversed(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise LocalDirectorySourceError(
            "local_root_invalid",
            "Local-directory root must be an existing canonical directory.",
        ) from error


def _verify_root_path(expected: _RootDescriptor) -> None:
    actual = _open_root(expected.path)
    try:
        if (actual.device, actual.inode) != (expected.device, expected.inode):
            raise LocalDirectorySourceError(
                "local_root_replaced",
                "Local-directory root identity changed after attachment.",
            )
    finally:
        os.close(actual.descriptor)


def _root_revision(root: _RootDescriptor) -> str:
    return (
        "sha256:"
        + sha256(
            canonical_json({"device": root.device, "inode": root.inode}).encode("utf-8")
        ).hexdigest()
    )


def _normalized_relative_path(value: str, source_id: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or len(value) > 2_048
    ):
        raise LocalDirectorySourceError(
            "resource_path_invalid",
            "The catalog file identity is not a normalized relative path.",
            source_id=source_id,
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(not _safe_segment(part) for part in path.parts)
    ):
        raise LocalDirectorySourceError(
            "resource_path_invalid",
            "The catalog file identity is not a normalized relative path.",
            source_id=source_id,
        )
    return value


def _read_configuration(
    registration: SourceRegistration,
) -> tuple[_DiscoveryLimits, Path, int, int]:
    values = registration.configuration
    root = values.get("root")
    root_device = values.get("root_device")
    root_inode = values.get("root_inode")
    formats = values.get("formats")
    if (
        not isinstance(root, str)
        or not root
        or registration.native_identity != root
        or not isinstance(root_device, int)
        or isinstance(root_device, bool)
        or root_device < 0
        or not isinstance(root_inode, int)
        or isinstance(root_inode, bool)
        or root_inode < 0
        or formats != ("csv", "json")
    ):
        raise LocalDirectorySourceError(
            "source_configuration_invalid",
            "The local-directory source configuration is invalid.",
            source_id=registration.id,
        )
    names_and_maxima = (
        ("max_depth", 64),
        ("max_files", 10_000),
        ("max_file_bytes", 2 * 1024 * 1024),
        ("max_columns", 4_096),
        ("max_rows", 1_000_000),
        ("max_json_nodes", 2_000_000),
        ("max_json_depth", 128),
        ("max_key_bytes", 64 * 1024),
        ("max_string_bytes", 8 * 1024 * 1024),
        ("max_cell_bytes", 8 * 1024 * 1024),
    )
    bounded: dict[str, int] = {}
    for name, maximum in names_and_maxima:
        value = values.get(name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 1 <= value <= maximum
        ):
            raise LocalDirectorySourceError(
                "source_configuration_invalid",
                "The local-directory source configuration is invalid.",
                source_id=registration.id,
            )
        bounded[name] = value
    return (
        _DiscoveryLimits(**bounded),
        Path(root),
        root_device,
        root_inode,
    )


def _current_file_facets(
    facets: tuple[CatalogFacet, ...],
    source_id: str,
) -> tuple[CatalogFacet, CatalogFacet]:
    files = tuple(item for item in facets if item.kind is FacetKind.FILE)
    tabular = tuple(item for item in facets if item.kind is FacetKind.TABULAR)
    if len(files) != 1 or len(tabular) != 1:
        raise LocalDirectorySourceError(
            "catalog_facets_invalid",
            "The file requires one current file facet and one tabular facet.",
            source_id=source_id,
        )
    return files[0], tabular[0]


def _expected_file_facts(
    file_facet: CatalogFacet,
    tabular_facet: CatalogFacet,
    source_revision: str | None,
    source_id: str,
) -> _ExpectedFileFacts:
    file_payload = file_facet.payload
    file_format = file_payload.get("format")
    media_type = file_payload.get("media_type")
    encoding = file_payload.get("encoding")
    size_bytes = file_payload.get("size_bytes")
    content_sha256 = file_payload.get("content_sha256")
    raw_columns = tabular_facet.payload.get("columns")
    if (
        file_format not in {"csv", "json"}
        or media_type != ("text/csv" if file_format == "csv" else "application/json")
        or encoding not in {"utf-8", "utf-8-sig"}
        or not isinstance(size_bytes, int)
        or isinstance(size_bytes, bool)
        or size_bytes < 0
        or not _canonical_sha256(content_sha256)
        or source_revision != content_sha256
        or not isinstance(raw_columns, tuple)
    ):
        raise LocalDirectorySourceError(
            "catalog_facets_invalid",
            "The current file catalog facts are incomplete or inconsistent.",
            source_id=source_id,
        )
    columns: list[str] = []
    for raw_column in raw_columns:
        if not isinstance(raw_column, Mapping):
            raise LocalDirectorySourceError(
                "catalog_facets_invalid",
                "The current file catalog facts are incomplete or inconsistent.",
                source_id=source_id,
            )
        name = raw_column.get("name")
        if not isinstance(name, str) or not name:
            raise LocalDirectorySourceError(
                "catalog_facets_invalid",
                "The current file catalog facts are incomplete or inconsistent.",
                source_id=source_id,
            )
        columns.append(name)
    if len(columns) != len({item.casefold() for item in columns}):
        raise LocalDirectorySourceError(
            "catalog_facets_invalid",
            "The current file catalog facts are incomplete or inconsistent.",
            source_id=source_id,
        )
    assert isinstance(file_format, str)
    assert isinstance(media_type, str)
    assert isinstance(encoding, str)
    assert isinstance(content_sha256, str)
    return _ExpectedFileFacts(
        format=file_format,
        media_type=media_type,
        encoding=encoding,
        size_bytes=size_bytes,
        content_sha256=content_sha256,
        columns=tuple(columns),
    )


def _canonical_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 71
        and value.startswith("sha256:")
        and all(character in "0123456789abcdef" for character in value[7:])
    )


def _read_registered_file(
    root_path: Path,
    root_device: int,
    root_inode: int,
    relative_path: str,
    max_bytes: int,
    source_id: str,
    cancellation: threading.Event,
) -> tuple[bytes, os.stat_result]:
    root = _open_root(root_path)
    descriptor = root.descriptor
    try:
        if (root.device, root.inode) != (root_device, root_inode):
            raise LocalDirectorySourceError(
                "local_root_replaced",
                "Local-directory root identity changed after attachment.",
                source_id=source_id,
            )
        parts = PurePosixPath(relative_path).parts
        for part in parts[:-1]:
            _check_cancelled(cancellation)
            before = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(before.st_mode) or stat.S_ISLNK(before.st_mode):
                raise OSError("path parent is not a real directory")
            child = _open_checked_directory(descriptor, part, before)
            os.close(descriptor)
            descriptor = child
        _check_cancelled(cancellation)
        name = parts[-1]
        before = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
            raise OSError("path target is not a regular file")
        return _read_checked_file(
            descriptor,
            name,
            before,
            max_bytes,
            source_id,
            cancellation,
        )
    finally:
        os.close(descriptor)


def _discover(
    root_descriptor: int,
    root: _RootDescriptor,
    registration: SourceRegistration,
    request: DiscoveryRequest,
    limits: _DiscoveryLimits,
    completed_at: datetime,
    cancellation: threading.Event,
) -> SourceCatalogSnapshot:
    try:
        opened = os.fstat(root_descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (
            int(opened.st_dev),
            int(opened.st_ino),
        ) != (root.device, root.inode):
            raise LocalDirectorySourceError(
                "local_root_replaced",
                "Local-directory root identity changed after attachment.",
                source_id=registration.id,
            )
        files: list[tuple[str, bytes, os.stat_result]] = []
        _scan_directory(
            root_descriptor,
            (),
            depth=0,
            files=files,
            limits=limits,
            source_id=registration.id,
            cancellation=cancellation,
        )
        files.sort(key=lambda item: item[0].encode("utf-8"))
        if len(files) + 1 > request.max_resources:
            raise DiscoveryLimitError(
                registration.id,
                "Local-directory discovery resource limit exceeded",
            )
        if len(files) > request.max_relationships:
            raise DiscoveryLimitError(
                registration.id,
                "Local-directory discovery relationship limit exceeded",
            )
        if completed_at < request.requested_at:
            completed_at = request.requested_at
        return _build_snapshot(
            registration,
            request,
            files,
            limits,
            completed_at,
            cancellation,
        )
    except LocalDirectorySourceError:
        raise
    except (
        csv.Error,
        json.JSONDecodeError,
        RecursionError,
        UnicodeError,
        ValueError,
    ) as error:
        raise LocalDirectorySourceError(
            "local_file_invalid",
            "A local tabular file violates the bounded CSV/JSON contract.",
            source_id=registration.id,
        ) from error
    except OSError as error:
        raise LocalDirectorySourceError(
            "local_discovery_failed",
            "Local-directory discovery could not complete safely.",
            source_id=registration.id,
        ) from error
    finally:
        os.close(root_descriptor)


def _scan_directory(
    directory_descriptor: int,
    parents: tuple[str, ...],
    *,
    depth: int,
    files: list[tuple[str, bytes, os.stat_result]],
    limits: _DiscoveryLimits,
    source_id: str,
    cancellation: threading.Event,
) -> None:
    _check_cancelled(cancellation)
    names = tuple(
        sorted(
            (name for name in os.listdir(directory_descriptor) if _safe_segment(name)),
            key=lambda name: name.encode("utf-8"),
        )
    )
    for name in names:
        _check_cancelled(cancellation)
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            continue
        if stat.S_ISDIR(before.st_mode):
            if depth >= limits.max_depth:
                continue
            child = _open_checked_directory(directory_descriptor, name, before)
            try:
                _scan_directory(
                    child,
                    (*parents, name),
                    depth=depth + 1,
                    files=files,
                    limits=limits,
                    source_id=source_id,
                    cancellation=cancellation,
                )
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(before.st_mode):
            continue
        suffix = PurePosixPath(name).suffix.casefold()
        if suffix not in _SUPPORTED_SUFFIXES:
            continue
        if len(files) >= limits.max_files:
            raise DiscoveryLimitError(
                source_id,
                "Local-directory discovery file limit exceeded",
            )
        content, after = _read_checked_file(
            directory_descriptor,
            name,
            before,
            limits.max_file_bytes,
            source_id,
            cancellation,
        )
        relative = PurePosixPath(*parents, name).as_posix()
        files.append((relative, content, after))


def _open_checked_directory(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
) -> int:
    descriptor = os.open(name, _root_flags(), dir_fd=parent_descriptor)
    try:
        after = os.fstat(descriptor)
        if not stat.S_ISDIR(after.st_mode) or not _same_identity(before, after):
            raise OSError("directory changed during traversal")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _read_checked_file(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
    max_bytes: int,
    source_id: str,
    cancellation: threading.Event,
) -> tuple[bytes, os.stat_result]:
    if before.st_size > max_bytes:
        raise DiscoveryLimitError(
            source_id,
            "Local-directory discovery file byte limit exceeded",
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_identity(before, opened):
            raise OSError("file changed before open")
        chunks: list[bytes] = []
        total = 0
        while True:
            _check_cancelled(cancellation)
            chunk = os.read(
                descriptor,
                min(_READ_CHUNK_BYTES, max_bytes + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise DiscoveryLimitError(
                    source_id,
                    "Local-directory discovery file byte limit exceeded",
                )
        finished = os.fstat(descriptor)
        if not _same_file_version(opened, finished):
            raise OSError("file changed during read")
        return b"".join(chunks), finished
    finally:
        os.close(descriptor)


def _safe_segment(name: object) -> bool:
    if not isinstance(name, str) or not name or name in {".", ".."}:
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    try:
        name.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    return True


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        int(left.st_dev),
        int(left.st_ino),
        stat.S_IFMT(left.st_mode),
    ) == (
        int(right.st_dev),
        int(right.st_ino),
        stat.S_IFMT(right.st_mode),
    )


def _same_file_version(left: os.stat_result, right: os.stat_result) -> bool:
    return _same_identity(left, right) and (
        int(left.st_size),
        int(left.st_mtime_ns),
        int(left.st_ctime_ns),
    ) == (
        int(right.st_size),
        int(right.st_mtime_ns),
        int(right.st_ctime_ns),
    )


def _check_cancelled(cancellation: threading.Event) -> None:
    if cancellation.is_set():
        raise RuntimeError("local discovery cancelled")


def _build_snapshot(
    registration: SourceRegistration,
    request: DiscoveryRequest,
    files: list[tuple[str, bytes, os.stat_result]],
    limits: _DiscoveryLimits,
    completed_at: datetime,
    cancellation: threading.Event,
) -> SourceCatalogSnapshot:
    folder_id = catalog_resource_id(registration.id, ResourceKind.FOLDER, ".")
    facts: list[_FileFacts] = []
    for relative_path, content, descriptor_stat in files:
        _check_cancelled(cancellation)
        resource_id = catalog_resource_id(
            registration.id,
            ResourceKind.FILE,
            relative_path,
        )
        content_hash = "sha256:" + sha256(content).hexdigest()
        suffix = PurePosixPath(relative_path).suffix.casefold()
        file_format = suffix.removeprefix(".")
        parsed = _parse_tabular(content, file_format, limits)
        modified_at = datetime.fromtimestamp(
            descriptor_stat.st_mtime,
            tz=timezone.utc,
        )
        file_facet = CatalogFacet.from_file(
            resource_id=resource_id,
            sync_id=request.sync_id,
            observed_at=request.requested_at,
            facet=FileFacet(
                format=file_format,
                media_type=("text/csv" if file_format == "csv" else "application/json"),
                encoding=parsed.encoding,
                size_bytes=len(content),
                content_sha256=content_hash,
                modified_at=modified_at,
            ),
        )
        tabular_facet = CatalogFacet.from_tabular(
            resource_id=resource_id,
            sync_id=request.sync_id,
            observed_at=request.requested_at,
            facet=TabularFacet(
                columns=tuple(
                    TabularColumn(
                        name=name,
                        native_type=("TEXT" if file_format == "csv" else "JSON"),
                        ordinal=index,
                        nullable=(
                            True
                            if file_format == "csv"
                            else any(
                                name not in row or row[name] is None
                                for row in parsed.rows
                            )
                        ),
                    )
                    for index, name in enumerate(parsed.columns)
                ),
                row_count_estimate=len(parsed.rows),
            ),
        )
        facts.append(
            _FileFacts(
                relative_path=relative_path,
                content_sha256=content_hash,
                file_facet=file_facet,
                tabular_facet=tabular_facet,
            )
        )

    relationships = tuple(
        CatalogRelationship.build(
            source_id=registration.id,
            from_resource_id=folder_id,
            to_resource_id=fact.file_facet.resource_id,
            kind=RelationshipKind.CONTAINS,
            provenance=RelationshipProvenance.CONNECTOR,
            confidence=1.0,
            sync_id=request.sync_id,
            observed_at=request.requested_at,
            attributes={"relative_path": fact.relative_path},
        )
        for fact in facts
    )
    manifest_revision = (
        "sha256:"
        + sha256(
            canonical_json(
                tuple(
                    {
                        "content_sha256": fact.content_sha256,
                        "relative_path": fact.relative_path,
                    }
                    for fact in facts
                )
            ).encode("utf-8")
        ).hexdigest()
    )

    folder_revision = CatalogResourceRevision.build(
        resource_id=folder_id,
        sync_id=request.sync_id,
        observed_at=request.requested_at,
        relationship_revisions=(item.revision for item in relationships),
        source_revision=manifest_revision,
    )
    resources: list[CatalogResource] = [
        CatalogResource.build(
            agent_id=request.agent_id,
            source_id=request.source_id,
            native_identity=".",
            external_uri=f"daita-local://{request.source_id}/",
            kind=ResourceKind.FOLDER,
            name=registration.display_name,
            sensitivity=Sensitivity.INTERNAL,
            revision=folder_revision,
            first_observed_at=request.requested_at,
            last_observed_at=request.requested_at,
        )
    ]
    revisions: list[CatalogResourceRevision] = [folder_revision]
    for fact, relationship in zip(facts, relationships, strict=True):
        revision = CatalogResourceRevision.build(
            resource_id=fact.file_facet.resource_id,
            sync_id=request.sync_id,
            observed_at=request.requested_at,
            facet_revisions=(
                fact.file_facet.revision,
                fact.tabular_facet.revision,
            ),
            relationship_revisions=(relationship.revision,),
            source_revision=fact.content_sha256,
        )
        revisions.append(revision)
        resources.append(
            CatalogResource.build(
                agent_id=request.agent_id,
                source_id=request.source_id,
                native_identity=fact.relative_path,
                external_uri=(
                    f"daita-local://{request.source_id}/"
                    f"{quote(fact.relative_path, safe='/')}"
                ),
                kind=ResourceKind.FILE,
                name=PurePosixPath(fact.relative_path).name,
                sensitivity=Sensitivity.INTERNAL,
                revision=revision,
                first_observed_at=request.requested_at,
                last_observed_at=request.requested_at,
            )
        )
    sync = CatalogSync(
        id=request.sync_id,
        agent_id=request.agent_id,
        source_id=request.source_id,
        adapter_id=_ADAPTER_ID,
        status=CatalogSyncStatus.SUCCEEDED,
        started_at=request.requested_at,
        completed_at=completed_at,
        source_revision=manifest_revision,
        resource_count=len(resources),
        relationship_count=len(relationships),
    )
    return SourceCatalogSnapshot(
        sync=sync,
        resources=tuple(resources),
        revisions=tuple(revisions),
        facets=tuple(
            facet for fact in facts for facet in (fact.file_facet, fact.tabular_facet)
        ),
        relationships=relationships,
    )


def _decode_utf8(content: bytes) -> tuple[str, str]:
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig", errors="strict"), "utf-8-sig"
    return content.decode("utf-8", errors="strict"), "utf-8"


def _parse_tabular(
    content: bytes,
    file_format: str,
    limits: _DiscoveryLimits,
) -> _ParsedFile:
    text, encoding = _decode_utf8(content)
    if file_format == "csv":
        columns, rows = _parse_csv(text, limits)
    elif file_format == "json":
        columns, rows = _parse_json(text, limits)
    else:
        raise ValueError("unsupported local tabular format")
    return _ParsedFile(
        format=file_format,
        encoding=encoding,
        columns=columns,
        rows=rows,
    )


def _parse_csv(
    text: str,
    limits: _DiscoveryLimits,
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, None)
    if header is None or not header:
        raise ValueError("CSV requires a header")
    if len(header) > limits.max_columns:
        raise ValueError("CSV column limit exceeded")
    for value in header:
        _validate_key(value, limits)
    if len(header) != len({value.casefold() for value in header}):
        raise ValueError("CSV column names must be unique")
    rows: list[dict[str, object]] = []
    for row in reader:
        if len(rows) >= limits.max_rows:
            raise ValueError("CSV row limit exceeded")
        if len(row) != len(header):
            raise ValueError("CSV rows must match the header")
        for value in row:
            _validate_string(value, limits)
            if _utf8_bytes(value) > limits.max_cell_bytes:
                raise ValueError("CSV cell limit exceeded")
        rows.append(dict(zip(header, row, strict=True)))
    return tuple(header), tuple(rows)


def _parse_json(
    text: str,
    limits: _DiscoveryLimits,
) -> tuple[tuple[str, ...], tuple[dict[str, object], ...]]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON value is forbidden: {value}")

    def reject_duplicate_keys(
        pairs: list[tuple[str, object]],
    ) -> dict[str, object]:
        result: dict[str, object] = {}
        folded: set[str] = set()
        for key, item in pairs:
            normalized = key.casefold()
            if normalized in folded:
                raise ValueError("JSON object keys must be unique")
            folded.add(normalized)
            result[key] = item
        return result

    value = json.loads(
        text,
        object_pairs_hook=reject_duplicate_keys,
        parse_constant=reject_constant,
    )
    _validate_json_value(value, limits)
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError("JSON tabular files must be an array of objects")
    if len(value) > limits.max_rows:
        raise ValueError("JSON row limit exceeded")
    names_by_folded: dict[str, str] = {}
    for row in value:
        assert isinstance(row, dict)
        for name, cell in row.items():
            _validate_key(name, limits)
            folded = name.casefold()
            previous = names_by_folded.get(folded)
            if previous is not None and previous != name:
                raise ValueError("JSON column names must be casefold unique")
            names_by_folded[folded] = name
            if len(names_by_folded) > limits.max_columns:
                raise ValueError("JSON column limit exceeded")
            if len(canonical_json(cell).encode("utf-8")) > limits.max_cell_bytes:
                raise ValueError("JSON cell limit exceeded")
    ordered = tuple(
        sorted(names_by_folded.values(), key=lambda item: item.encode("utf-8"))
    )
    return ordered, tuple(value)


def _validate_json_value(value: object, limits: _DiscoveryLimits) -> None:
    pending = [(value, 1)]
    seen = 0
    while pending:
        current, depth = pending.pop()
        if depth > limits.max_json_depth:
            raise ValueError("JSON depth limit exceeded")
        seen += 1
        if seen > limits.max_json_nodes:
            raise ValueError("JSON node limit exceeded")
        if isinstance(current, dict):
            for key, item in current.items():
                _validate_key(key, limits)
                pending.append((item, depth + 1))
            seen += len(current)
            if seen > limits.max_json_nodes:
                raise ValueError("JSON node limit exceeded")
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
        elif isinstance(current, str):
            _validate_string(current, limits)
        elif isinstance(current, float) and not math.isfinite(current):
            raise ValueError("JSON numbers must be finite")
        elif current is not None and not isinstance(current, (bool, int, float)):
            raise ValueError("JSON value type is unsupported")


def _validate_key(value: object, limits: _DiscoveryLimits) -> None:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x00" in value
        or len(value) > 256
        or _utf8_bytes(value) > limits.max_key_bytes
    ):
        raise ValueError("tabular keys must be bounded non-empty strings")


def _validate_string(value: str, limits: _DiscoveryLimits) -> None:
    if "\x00" in value or _utf8_bytes(value) > limits.max_string_bytes:
        raise ValueError("tabular string limit exceeded")


def _utf8_bytes(value: str) -> int:
    return len(value.encode("utf-8", errors="strict"))


__all__ = [
    "LocalDirectoryReadBackend",
    "LocalDirectoryResourceAdapter",
    "LocalDirectorySource",
    "LocalDirectorySourceError",
]
