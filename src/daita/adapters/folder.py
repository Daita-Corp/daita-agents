"""Folder sources: cataloged attached folders and the live session folder.

An attached folder is a durable source like any other: discovery catalogs its
folders and files, and reads are bound to the exact catalog revisions. The
session folder is the host's per-session working directory: it is never stored
or cataloged, and its search, read and query operations run live. Both use the
descriptor-contained ``LocalWorkspaceBackend`` and its private DuckDB worker.
"""

from __future__ import annotations

import asyncio
import os
import stat
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, cast

from .._json import FrozenJsonObject, canonical_json
from ..catalog.models import (
    CatalogEntry,
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogTier,
    FileFacet,
    RelationshipKind,
    RelationshipProvenance,
    ResourceBatch,
    ResourceKind,
    Sensitivity,
    TabularColumn,
    TabularFacet,
    catalog_resource_id,
)
from ..errors import DaitaError
from ..llm.models import ModelSensitivity
from ..storage.sqlite_records import SourceReadMode
from ..workspace import LocalFileAccess, LocalWorkspace
from .local_file_query import (
    LocalFileQueryError,
    LocalFileQueryLimits,
    LocalFileQueryResult,
)
from .local_workspace import (
    LocalWorkspaceBackend,
    LocalWorkspaceError,
    LocalWorkspaceLimits,
    physical_revision_for_facts,
    structured_file_format,
)
from .models import (
    DiscoveryRequest,
    ResourceRef,
    ResourceSnapshot,
    SourceHealth,
    SourceRegistration,
)
from .protocols import (
    DESCRIBE_MAX_COLUMNS,
    DESCRIBE_MAX_LISTED_TABLES,
    ResourceAdapterError,
    ResourceNotFoundError,
    SourceClosedError,
    latest_snapshot,
)

if TYPE_CHECKING:
    from ..domains.data.capabilities import SqlReadResult
    from ..domains.data.source_files import (
        FileSearchRequest,
        PathPatternQueryRequest,
        SourceFileResult,
        TextReadRequest,
    )
    from ..domains.data.sql import CheckedRead
    from .registry import SourceBackendContext

FOLDER_ADAPTER_ID = "folder"
_UNREADABLE_PATH_CODES = frozenset(
    {
        "path_not_allowed",
        "path_restricted",
        "path_invalid",
        "symlink_not_allowed",
        "not_regular_file",
    }
)
_DESCRIBE_BATCH_FILES = 32
# The walk and the schema reads each get this budget; a discovery that runs
# out of either commits a snapshot marked truncated.
_DISCOVERY_SECONDS = 60.0
_RUN_ID = "folder-source"


class FolderSourceError(DaitaError):
    """Stable model-visible failure of a folder-source read."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message, error_code=code)


def session_folder_source_id(agent_id: str, workspace: LocalWorkspace) -> str:
    """The stable identity of one agent's live session folder source.

    Its ``source:workspace:sha256:`` form can never be a stored registration and
    sorts after every durable ``source:sha256:`` identity.
    """

    material = canonical_json(
        {
            "access": workspace.access.value,
            "agent_id": agent_id,
            "root": str(workspace.root),
        }
    )
    return "source:workspace:sha256:" + sha256(material.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SessionFolderSource:
    """The in-memory routing facts of one session's live folder source."""

    source_id: str
    display_name: str
    summary: str
    presentation_sensitivity: ModelSensitivity
    adapter_id: str = FOLDER_ADAPTER_ID

    @classmethod
    def for_workspace(
        cls, agent_id: str, workspace: LocalWorkspace
    ) -> SessionFolderSource:
        computer = workspace.access is LocalFileAccess.COMPUTER
        return cls(
            source_id=session_folder_source_id(agent_id, workspace),
            display_name="This computer" if computer else "Working folder",
            summary="Local files on this computer." if computer else "Local files.",
            presentation_sensitivity=workspace.sensitivity,
        )

    def routing_fact(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "source_id": self.source_id,
                "adapter_id": self.adapter_id,
                "display_name": self.display_name,
                "summary": self.summary,
                "when_to_use": "",
                "keywords": (),
                "presentation_sensitivity": self.presentation_sensitivity.value,
            }
        )


@dataclass(frozen=True, slots=True)
class FolderSource:
    """A durable folder source, always contained to its root."""

    root: str | Path
    subpaths: tuple[str, ...] = ()
    name: str | None = None
    sensitivity: ModelSensitivity = ModelSensitivity.INTERNAL

    def __post_init__(self) -> None:
        if not isinstance(self.subpaths, tuple) or any(
            not isinstance(item, str) or not item for item in self.subpaths
        ):
            raise ValueError("folder subpaths must be a tuple of relative paths")
        if self.name is not None and (not isinstance(self.name, str) or not self.name):
            raise ValueError("folder name must be non-empty text or None")

    def workspace(self) -> LocalWorkspace:
        return LocalWorkspace(
            Path(self.root),
            sensitivity=self.sensitivity,
            access=LocalFileAccess.WORKSPACE,
        )

    async def open(
        self,
        *,
        agent_id: str,
        attached_at: datetime,
        clock: Callable[[], datetime],
    ) -> FolderResourceAdapter:
        workspace = self.workspace()
        registration = SourceRegistration.build(
            agent_id=agent_id,
            adapter_id=FOLDER_ADAPTER_ID,
            native_identity=str(workspace.root),
            display_name=self.name or workspace.root.name,
            configuration={
                "root": str(workspace.root),
                "subpaths": list(self.subpaths),
            },
            attached_at=attached_at,
            presentation_sensitivity=self.sensitivity,
        )
        return FolderResourceAdapter(
            registration=registration,
            workspace=workspace,
            subpaths=self.subpaths,
            clock=clock,
        )


def folder_source_from_registration(
    registration: SourceRegistration,
    *,
    secret_provider: object = None,
) -> FolderSource:
    """Reconstruct one attached folder from its exact stored configuration."""

    del secret_provider
    configuration = registration.configuration
    if set(configuration) != {"root", "subpaths"}:
        raise ValueError("source configuration fields are invalid")
    root = configuration["root"]
    if not isinstance(root, str) or not root:
        raise ValueError("source configuration text is invalid")
    return FolderSource(
        root=root,
        subpaths=cast(tuple[str, ...], configuration["subpaths"]),
        name=registration.display_name,
        sensitivity=registration.presentation_sensitivity,
    )


class FolderResourceAdapter:
    def __init__(
        self,
        *,
        registration: SourceRegistration,
        workspace: LocalWorkspace,
        subpaths: tuple[str, ...],
        clock: Callable[[], datetime],
    ) -> None:
        self._registration = registration
        self._workspace = workspace
        self._subpaths = subpaths
        self._clock = clock
        self._lock = asyncio.Lock()
        self._closed = False
        self._latest: ResourceBatch | None = None

    @property
    def registration(self) -> SourceRegistration:
        return self._registration

    async def reconcile(
        self, request: DiscoveryRequest
    ) -> AsyncIterator[ResourceBatch]:
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
        async with self._lock:
            self._require_open()
            if request.agent_root is None or request.agent_home is None:
                raise ResourceAdapterError(
                    self._registration.id,
                    "folder_state_unavailable",
                    "Folder discovery needs the host's private agent state paths.",
                )
            try:
                backend = await LocalWorkspaceBackend.open(
                    self._workspace,
                    agent_root=request.agent_root,
                    agent_home=request.agent_home,
                    limits=LocalWorkspaceLimits(
                        max_search_results=request.max_inventory,
                        max_search_entries=1_000_000,
                        max_search_seconds=_DISCOVERY_SECONDS,
                    ),
                    clock=self._clock,
                )
            except LocalWorkspaceError as error:
                raise ResourceAdapterError(
                    self._registration.id, error.code, error.message
                ) from error
            try:
                batches = await _discover(
                    backend, self._registration, self._subpaths, request
                )
            except LocalWorkspaceError as error:
                raise ResourceAdapterError(
                    self._registration.id, error.code, error.message
                ) from error
            finally:
                await backend.close()
            self._latest = batches[1]
        for batch in batches:
            yield batch

    async def inspect(self, resource: ResourceRef) -> ResourceSnapshot:
        if not isinstance(resource, ResourceRef):
            raise TypeError("resource must be a ResourceRef")
        async with self._lock:
            self._require_open()
            if (
                resource.agent_id != self._registration.agent_id
                or resource.source_id != self._registration.id
            ):
                raise ResourceNotFoundError(self._registration.id, resource.resource_id)
            return latest_snapshot(self._latest, resource, self._clock())

    async def health(self) -> SourceHealth:
        async with self._lock:
            checked_at = self._clock()
            revision = _root_revision(self._workspace.root)
            return SourceHealth(
                agent_id=self._registration.agent_id,
                source_id=self._registration.id,
                adapter_id=FOLDER_ADAPTER_ID,
                healthy=not self._closed and revision is not None,
                checked_at=checked_at,
                source_revision=revision,
                error_code=(
                    "source_closed"
                    if self._closed
                    else ("folder_unavailable" if revision is None else None)
                ),
                details={"read_only": True},
            )

    async def close(self) -> None:
        async with self._lock:
            self._closed = True

    def _require_open(self) -> None:
        if self._closed:
            raise SourceClosedError(self._registration.id)


def _root_revision(root: Path) -> str | None:
    """The stat-based revision of the folder root, or None if it is gone."""

    try:
        facts = os.stat(root, follow_symlinks=False)
    except OSError:
        return None
    return physical_revision_for_facts(facts) if stat.S_ISDIR(facts.st_mode) else None


# Attached folders catalog only data files, one media type per query format.
_DATA_MEDIA_TYPES = {
    "csv": "text/csv",
    "tsv": "text/tab-separated-values",
    "json_records": "application/json",
    "ndjson": "application/x-ndjson",
    "parquet": "application/vnd.apache.parquet",
}


async def _discover(
    backend: LocalWorkspaceBackend,
    registration: SourceRegistration,
    subpaths: tuple[str, ...],
    request: DiscoveryRequest,
) -> tuple[ResourceBatch, ResourceBatch]:
    """Catalog the data files and the folders that hold them, bounded by the
    request, through the scanner: every one as inventory, the first
    ``max_resources`` described. Other files stay reachable live."""

    root_revision = _root_revision(backend.working_directory)
    if root_revision is None:
        raise LocalWorkspaceError(
            "folder_unavailable", "The folder root is unavailable."
        )
    result = await backend.search(
        run_id=_RUN_ID,
        paths=subpaths or None,
        mode="paths",
        glob="*",
        order_by="path",
        structured_only=True,
    )
    inventory_truncated = result.truncated
    folders: list[str] = ["."]
    seen_folders = {"."}
    files: list[tuple[str, str, int, datetime]] = []
    # Detail covers a prefix of the walk; the inventory continues past it.
    detail_folders, detail_files = 1, 0
    detail_cut = False
    for match in result.matches:
        parents = [
            parent.as_posix()
            for parent in reversed(PurePosixPath(match.path).parents)
            if parent.as_posix() != "."
        ]
        missing = [item for item in parents if item not in seen_folders]
        if len(folders) + len(files) + len(missing) + 1 > request.max_inventory:
            inventory_truncated = True
            break
        # One resource per file and missing folder; one CONTAINS edge each.
        needed = detail_folders + detail_files + len(missing) + 1
        if not detail_cut and (
            needed > request.max_resources or needed - 1 > request.max_relationships
        ):
            detail_cut = True
        if not detail_cut:
            detail_folders += len(missing)
            detail_files += 1
        folders.extend(missing)
        seen_folders.update(missing)
        files.append(
            (
                match.path,
                match.physical_revision,
                match.size_bytes,
                datetime.fromisoformat(match.modified_at),
            )
        )
    inventory_entries = tuple(
        request.inventory_entry(
            kind=kind,
            native_identity=path,
            name=registration.display_name if path == "." else path,
            external_uri=(backend.working_directory / path).as_uri(),
            sensitivity=Sensitivity(registration.presentation_sensitivity.value),
        )
        for kind, path in (
            *((ResourceKind.FOLDER, item) for item in folders),
            *((ResourceKind.FILE, item[0]) for item in files),
        )
    )
    inventory = ResourceBatch(
        tier=CatalogTier.INVENTORY,
        entries=inventory_entries,
        truncated=inventory_truncated,
    )
    folders = folders[:detail_folders]
    files = files[:detail_files]
    truncated = detail_cut
    schemas, described_all = await _describe(backend, tuple(path for path, *_ in files))
    # A data file left without its schema is an incomplete catalog.
    truncated = truncated or not described_all

    sync_id = request.sync_id
    observed_at = request.requested_at
    source_id = registration.id
    sensitivity = Sensitivity(registration.presentation_sensitivity.value)

    def resource_id(kind: ResourceKind, path: str) -> str:
        return catalog_resource_id(source_id, kind, path)

    def parent_of(path: str) -> str:
        return PurePosixPath(path).parent.as_posix()

    relationships = tuple(
        CatalogRelationship.build(
            source_id=source_id,
            from_resource_id=resource_id(ResourceKind.FOLDER, parent_of(path)),
            to_resource_id=resource_id(kind, path),
            kind=RelationshipKind.CONTAINS,
            provenance=RelationshipProvenance.CONNECTOR,
            confidence=1.0,
            sync_id=sync_id,
            observed_at=observed_at,
        )
        for kind, path in (
            *((ResourceKind.FOLDER, item) for item in folders[1:]),
            *((ResourceKind.FILE, path) for path, *_ in files),
        )
    )
    facets: list[CatalogFacet] = []
    entries: list[tuple[ResourceKind, str, tuple[str, ...], str | None]] = []
    for path in folders:
        entries.append((ResourceKind.FOLDER, path, (), None))
    for path, physical_revision, size_bytes, modified_at in files:
        identifier = resource_id(ResourceKind.FILE, path)
        format_name = cast(str, structured_file_format(path))
        file_facet = CatalogFacet.from_file(
            resource_id=identifier,
            sync_id=sync_id,
            observed_at=observed_at,
            facet=FileFacet(
                format=format_name,
                media_type=_DATA_MEDIA_TYPES[format_name],
                encoding=None if format_name == "parquet" else "utf-8",
                size_bytes=size_bytes,
                physical_revision=physical_revision,
                modified_at=modified_at,
            ),
        )
        own = [file_facet]
        columns = schemas.get(path)
        if columns and len(columns) <= request.max_columns_per_resource:
            own.append(
                CatalogFacet.from_tabular(
                    resource_id=identifier,
                    sync_id=sync_id,
                    observed_at=observed_at,
                    facet=TabularFacet(
                        columns=tuple(
                            TabularColumn(
                                name=name,
                                native_type=native_type,
                                ordinal=ordinal,
                                nullable=True,
                            )
                            for ordinal, (name, native_type) in enumerate(columns)
                        )
                    ),
                )
            )
        facets.extend(own)
        entries.append(
            (
                ResourceKind.FILE,
                path,
                tuple(item.revision for item in own),
                physical_revision,
            )
        )
    relationship_revisions: dict[str, list[str]] = {}
    for item in relationships:
        for endpoint in (item.from_resource_id, item.to_resource_id):
            relationship_revisions.setdefault(endpoint, []).append(item.revision)
    facets_by_resource: dict[str, list[CatalogFacet]] = {}
    for facet in facets:
        facets_by_resource.setdefault(facet.resource_id, []).append(facet)
    detailed: list[CatalogEntry] = []
    for kind, path, facet_revisions, revision_source in entries:
        identifier = resource_id(kind, path)
        revision = CatalogResourceRevision.build(
            resource_id=identifier,
            sync_id=sync_id,
            observed_at=observed_at,
            facet_revisions=facet_revisions,
            relationship_revisions=tuple(relationship_revisions.get(identifier, ())),
            source_revision=revision_source or root_revision,
        )
        detailed.append(
            CatalogEntry(
                resource=CatalogResource.build(
                    agent_id=request.agent_id,
                    source_id=source_id,
                    native_identity=path,
                    external_uri=(backend.working_directory / path).as_uri(),
                    kind=kind,
                    name=registration.display_name if path == "." else path,
                    sensitivity=sensitivity,
                    revision=revision,
                    first_observed_at=observed_at,
                    last_observed_at=observed_at,
                ),
                revision=revision,
                facets=tuple(facets_by_resource.get(identifier, ())),
                detailed=True,
            )
        )
    detail = ResourceBatch(
        tier=CatalogTier.DETAIL,
        entries=tuple(detailed),
        relationships=relationships,
        truncated=truncated,
        source_revision=root_revision,
    )
    return inventory, detail


async def _describe(
    backend: LocalWorkspaceBackend, paths: tuple[str, ...]
) -> tuple[dict[str, tuple[tuple[str, str], ...]], bool]:
    """Read each structured file's schema, one worker per format batch.

    Also returns whether every batch was attempted within the budget.
    """

    by_format: dict[str, list[str]] = {}
    for path in paths:
        format_name = structured_file_format(path)
        if format_name is not None:
            by_format.setdefault(format_name, []).append(path)
    deadline = time.monotonic() + _DISCOVERY_SECONDS
    described: dict[str, tuple[tuple[str, str], ...]] = {}
    pending = [
        tuple(group[index : index + _DESCRIBE_BATCH_FILES])
        for group in by_format.values()
        for index in range(0, len(group), _DESCRIBE_BATCH_FILES)
    ]
    while pending and time.monotonic() < deadline:
        batch = pending.pop(0)
        try:
            schemas = await backend.describe(run_id=_RUN_ID, paths=batch)
        except (LocalWorkspaceError, LocalFileQueryError):
            # One unreadable file fails its batch; retry its files singly.
            if len(batch) > 1:
                pending[:0] = [(item,) for item in batch]
            continue
        for path, schema in zip(batch, schemas, strict=True):
            if schema:
                described[path] = schema
    return described, not pending


class FolderBackend:
    """Serve every folder operation for the session and attached folders."""

    def __init__(self, context: SourceBackendContext) -> None:
        self._store = context.store
        self._home = context.home
        self._session = context.session_workspace
        self._session_source_id = context.session_source_id
        self._clock = context.clock
        self._attached: dict[str, LocalWorkspaceBackend] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    def _live(self, source_id: str) -> LocalWorkspaceBackend | None:
        if source_id != self._session_source_id:
            return None
        if self._session is None:
            raise LocalWorkspaceError(
                "workspace_unavailable", "Local file access is unavailable."
            )
        return self._session

    async def _attached_backend(
        self, agent_id: str, source_id: str
    ) -> tuple[LocalWorkspaceBackend, FolderSource]:
        registration = await self._store.load_source(agent_id, source_id)
        if (
            registration is None
            or registration.detached_at is not None
            or registration.adapter_id != FOLDER_ADAPTER_ID
            or self._home is None
        ):
            raise LocalWorkspaceError(
                "workspace_unavailable", "The folder source is unavailable."
            )
        source = folder_source_from_registration(registration)
        async with self._lock:
            if self._closed:
                raise LocalWorkspaceError(
                    "workspace_unavailable", "The folder source is closed."
                )
            backend = self._attached.get(source_id)
            if backend is None or backend.closed:
                backend = await LocalWorkspaceBackend.open(
                    source.workspace(),
                    agent_root=self._home.parent.parent,
                    agent_home=self._home,
                    clock=self._clock,
                )
                self._attached[source_id] = backend
        return backend, source

    async def _path_admission(
        self,
        agent_id: str,
        source_id: str,
        source: FolderSource,
        narrowed: frozenset[str] | None,
    ) -> Callable[[str], bool]:
        """Decide from each file itself: a data file, in the folder's subpaths, and
        readable now. Current permission is read here because a call can queue after
        it was prepared; ``narrowed`` is the run's own limit, None when it has none.
        """

        scope = await self._store.load_source_read_scope(agent_id, source_id)
        if scope is None or scope.mode is SourceReadMode.NONE:
            return lambda _path: False
        selected = (
            None if scope.mode is SourceReadMode.ALL else frozenset(scope.resource_ids)
        )

        def admitted(path: str) -> bool:
            if structured_file_format(path) is None or not _within_subpaths(
                path, source.subpaths
            ):
                return False
            resource_id = _file_resource_id(source_id, path)
            return all(
                limit is None or resource_id in limit for limit in (selected, narrowed)
            )

        return admitted

    def _result(
        self,
        source_id: str,
        data: FrozenJsonObject,
        provenance: FrozenJsonObject,
        sensitivity: ModelSensitivity,
        **overrides: object,
    ) -> SourceFileResult:
        from ..domains.data.source_files import SourceFileResult

        return SourceFileResult(
            data=FrozenJsonObject.from_mapping(
                {"source_id": source_id, **data.to_dict(), **overrides}
            ),
            sensitivity=sensitivity,
            provenance=FrozenJsonObject.from_mapping(
                {**provenance.to_dict(), "source_id": source_id}
            ),
        )

    async def _lookup_backend(
        self, agent_id: str, source_id: str
    ) -> tuple[
        LocalWorkspaceBackend, Callable[[str], bool] | None, tuple[str, ...] | None
    ]:
        live = self._live(source_id)
        if live is not None:
            return live, None, None
        backend, source = await self._attached_backend(agent_id, source_id)
        admitted = await self._path_admission(agent_id, source_id, source, None)
        return backend, admitted, source.subpaths or None

    async def search_names(
        self, *, agent_id: str, source_id: str, words: tuple[str, ...], limit: int
    ) -> tuple[Mapping[str, object], ...]:
        """Match words against the current listing of this folder's data files."""

        backend, admitted, paths = await self._lookup_backend(agent_id, source_id)
        result = await backend.search(
            run_id=_RUN_ID,
            paths=paths,
            mode="paths",
            glob="*",
            order_by="path",
            admit_path=admitted,
            structured_only=True,
        )
        scored = (
            (sum(word in match.path.lower() for word in words), match.path)
            for match in result.matches
        )
        return tuple(
            {"name": path, "kind": "file", "columns": None, "row_estimate": None}
            for _, path in sorted(
                (item for item in scored if item[0]),
                key=lambda item: (-item[0], item[1]),
            )[:limit]
        )

    async def describe_name(
        self, *, agent_id: str, source_id: str, name: str
    ) -> Mapping[str, object] | None:
        """Read one data file's columns live, or list one subfolder's data files."""

        backend, admitted, _paths = await self._lookup_backend(agent_id, source_id)
        try:
            if structured_file_format(name) is not None:
                if admitted is not None and not admitted(name):
                    return None
                schema = (await backend.describe(run_id=_RUN_ID, paths=(name,)))[0]
                return {
                    "name": name,
                    "kind": "file",
                    "comment": None,
                    "row_estimate": None,
                    "partitions": 0,
                    "columns": [
                        {"name": column, "type": kind, "nullable": True}
                        for column, kind in (schema or ())[:DESCRIBE_MAX_COLUMNS]
                    ],
                    "columns_truncated": len(schema or ()) > DESCRIBE_MAX_COLUMNS,
                    "primary_key": [],
                    "unique_keys": [],
                    "relationships": [],
                }
            result = await backend.search(
                run_id=_RUN_ID,
                path=name,
                mode="paths",
                glob="*",
                order_by="path",
                admit_path=admitted,
                structured_only=True,
            )
        except (LocalWorkspaceError, LocalFileQueryError):
            return None
        if not result.matches:
            return None
        return {
            "name": name,
            "kind": "folder",
            "tables": [
                {"name": match.path, "kind": "file", "row_estimate": None}
                for match in result.matches[:DESCRIBE_MAX_LISTED_TABLES]
            ],
            "total_tables": len(result.matches),
        }

    async def search_files(self, request: FileSearchRequest) -> SourceFileResult:
        live = self._live(request.source_id)
        if live is not None:
            result = await live.search(
                run_id=request.run_id,
                query=request.query,
                path=request.path,
                paths=request.paths,
                mode=request.mode,
                glob=request.glob,
                order_by=request.order_by,
            )
            return self._result(
                request.source_id,
                result.to_mapping(),
                result.provenance_mapping(),
                live.sensitivity,
            )
        backend, source = await self._attached_backend(
            request.agent_id, request.source_id
        )
        admitted = await self._path_admission(
            request.agent_id, request.source_id, source, request.resource_ids
        )
        paths = request.paths
        if request.path is None and paths is None and source.subpaths:
            paths = source.subpaths
        result = await backend.search(
            run_id=request.run_id,
            query=request.query,
            path=request.path,
            paths=paths,
            mode=request.mode,
            glob=request.glob,
            order_by=request.order_by,
            admit_path=admitted,
        )
        return self._result(
            request.source_id,
            result.to_mapping(),
            result.provenance_mapping(),
            source.sensitivity,
        )

    async def read_text(self, request: TextReadRequest) -> SourceFileResult:
        live = self._live(request.source_id)
        if live is not None:
            result = await live.read(
                run_id=request.run_id,
                path=request.path,
                cursor=request.cursor,
                position=request.position,
            )
            return self._result(
                request.source_id,
                result.to_mapping(),
                result.provenance_mapping(),
                live.sensitivity,
            )
        backend, source = await self._attached_backend(
            request.agent_id, request.source_id
        )
        admitted = await self._path_admission(
            request.agent_id, request.source_id, source, request.resource_ids
        )
        try:
            result = await backend.read(
                run_id=request.run_id,
                path=request.path,
                cursor=request.cursor,
                position=request.position,
                admit_path=admitted,
            )
        except LocalWorkspaceError as error:
            if error.code in _UNREADABLE_PATH_CODES:
                raise _not_readable() from error
            raise
        # Cataloged folders are read-only sources: no edit binding.
        return self._result(
            request.source_id,
            result.to_mapping(),
            result.provenance_mapping(),
            source.sensitivity,
            binding=None,
        )

    async def query_path_pattern(
        self, request: PathPatternQueryRequest
    ) -> SourceFileResult:
        live = self._live(request.source_id)
        if live is not None:
            result = await live.query(
                run_id=request.run_id,
                path_pattern=request.path_pattern,
                canonical_sql=request.canonical_sql,
                sql_fingerprint=request.sql_fingerprint,
            )
            return self._result(
                request.source_id,
                result.data,
                result.sensitivity_provenance,
                live.sensitivity,
            )
        result, _paths, source = await self._query_attached(
            request.agent_id,
            request.source_id,
            request.path_pattern,
            request.canonical_sql,
            request.sql_fingerprint,
            request.resource_ids,
        )
        return self._result(
            request.source_id,
            result.data,
            result.sensitivity_provenance,
            source.sensitivity,
        )

    async def execute_read(
        self,
        *,
        agent_id: str,
        source_id: str,
        read: CheckedRead,
        max_rows: int,
        max_bytes: int,
        timeout_seconds: float | None = None,
    ) -> SqlReadResult:
        """Serve a combined-query input through the same attached-folder route."""

        from ..domains.data.capabilities import SqlReadResult
        from ..domains.data.results import BoundedResultProjection

        if read.path_pattern is None or read.parameters:
            raise FolderSourceError(
                "sql_invalid_input",
                "SQL over folder files takes path_pattern and no parameters.",
            )
        result, paths, _source = await self._query_attached(
            agent_id,
            source_id,
            read.path_pattern,
            read.canonical_sql,
            read.sql_fingerprint,
            (
                None
                if read.allowed_resource_ids is None
                else frozenset(read.allowed_resource_ids)
            ),
            LocalFileQueryLimits(
                max_result_rows=max_rows,
                max_result_bytes=max_bytes,
                max_query_seconds=min(30.0, timeout_seconds or 30.0),
            ),
        )
        data = result.data
        rows = cast(tuple[Mapping[str, object], ...], data["rows"])
        names = tuple((_file_resource_id(source_id, path), path) for path in paths)
        return SqlReadResult(
            source_id=source_id,
            canonical_sql=read.canonical_sql,
            sql_fingerprint=read.sql_fingerprint,
            resource_ids=tuple(item[0] for item in names),
            columns=cast(tuple[str, ...], data["columns"]),
            projection=BoundedResultProjection(
                rows=tuple(FrozenJsonObject.from_mapping(row) for row in rows),
                total_rows=cast(int, data["observed_rows"]),
                returned_rows=cast(int, data["returned_rows"]),
                utf8_bytes=cast(int, data["utf8_bytes"]),
                row_limit=cast(int, data["row_limit"]),
                byte_limit=cast(int, data["byte_limit"]),
                truncated=cast(bool, data["truncated"]),
                truncation_reasons=cast(tuple[str, ...], data["truncation_reasons"]),
            ),
            resource_names=names,
        )

    async def _query_attached(
        self,
        agent_id: str,
        source_id: str,
        path_pattern: str,
        canonical_sql: str,
        sql_fingerprint: str,
        narrowed: frozenset[str] | None,
        limits: LocalFileQueryLimits | None = None,
    ) -> tuple[LocalFileQueryResult, tuple[str, ...], FolderSource]:
        """Match the pattern against current files; each must be in this run's scope."""

        backend, source = await self._attached_backend(agent_id, source_id)
        admitted = await self._path_admission(agent_id, source_id, source, narrowed)
        manifest = await backend.bind_query_manifest(
            run_id=_RUN_ID, path_pattern=path_pattern
        )
        try:
            paths = tuple(binding.relative_path for binding in manifest.bindings)
            denied = [path for path in paths if not admitted(path)]
            if denied:
                raise FolderSourceError(
                    "resource_read_not_allowed",
                    f"{', '.join(denied[:8])} is not available for reading in "
                    "this run.",
                )
            result = await backend.query_manifest(
                manifest=manifest,
                canonical_sql=canonical_sql,
                sql_fingerprint=sql_fingerprint,
                limits=limits,
            )
        finally:
            manifest.close()
        return result, paths, source

    async def retire_source(self, agent_id: str, source_id: str) -> None:
        del agent_id
        async with self._lock:
            backend = self._attached.pop(source_id, None)
        if backend is not None:
            await backend.close()

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            backends = tuple(self._attached.values())
            self._attached.clear()
        for backend in backends:
            await backend.close()


def _within_subpaths(path: str, subpaths: tuple[str, ...]) -> bool:
    return not subpaths or any(
        path == item.rstrip("/") or path.startswith(item.rstrip("/") + "/")
        for item in subpaths
    )


def _file_resource_id(source_id: str, path: str) -> str:
    return catalog_resource_id(source_id, ResourceKind.FILE, path)


def _not_readable() -> FolderSourceError:
    return FolderSourceError(
        "resource_read_not_allowed",
        "The file is not a readable cataloged resource of this source.",
    )


__all__ = [
    "FolderBackend",
    "FolderResourceAdapter",
    "FolderSource",
    "FolderSourceError",
    "SessionFolderSource",
    "folder_source_from_registration",
    "session_folder_source_id",
]
