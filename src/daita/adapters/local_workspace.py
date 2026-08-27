"""Secure on-demand search and read access for one admitted local workspace."""

from __future__ import annotations

import asyncio
import base64
import codecs
import fnmatch
import hmac
import json
import mimetypes
import os
import re
import secrets
import stat
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path, PurePosixPath
from typing import Callable, Literal, TypeVar

from .._json import FrozenJsonObject, canonical_json
from ..workspace import LocalWorkspace, paths_overlap
from .local_file_query import (
    LocalFileQueryBackend,
    LocalFileQueryLimits,
    LocalFileQueryResult,
)

_READ_CHUNK_BYTES = 64 * 1_024
_MAX_LOGICAL_PATH_CHARACTERS = 2_048
_MAX_TOKEN_BYTES = 8 * 1_024
_MAX_QUERY_CHARACTERS = 512
_MAX_GLOB_CHARACTERS = 256
_MAX_EXCERPT_CHARACTERS = 320
_MAX_QUERY_PATTERN_CHARACTERS = 2_048
_LOCAL_QUERY_FORMATS = {
    ".csv": "csv",
    ".tsv": "tsv",
    ".json": "json_records",
    ".jsonl": "ndjson",
    ".ndjson": "ndjson",
    ".parquet": "parquet",
}
_SEARCH_NOISE_DIRECTORIES = frozenset(
    {
        ".git",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".svn",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
    }
)
_RESTRICTED_DIRECTORY_NAMES = frozenset({".aws", ".gnupg", ".ssh"})
_RESTRICTED_FILE_NAMES = frozenset(
    {
        ".netrc",
        "credentials",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "id_rsa",
    }
)
_RESTRICTED_FILE_SUFFIXES = frozenset({".key", ".p12", ".pem", ".pfx"})
_TOKEN_VERSION = "lw1"

_T = TypeVar("_T")


class LocalWorkspaceError(RuntimeError):
    """One normalized workspace failure that never includes the absolute root."""

    def __init__(
        self,
        code: str,
        message: str,
        details: dict[str, object] | None = None,
    ) -> None:
        if not isinstance(code, str) or not code:
            raise ValueError("workspace error code must be non-empty text")
        if not isinstance(message, str) or not message:
            raise ValueError("workspace error message must be non-empty text")
        self.code = code
        self.message = message
        self.details = FrozenJsonObject.from_mapping(details or {})
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class LocalWorkspaceLimits:
    max_search_results: int = 100
    max_search_entries: int = 50_000
    max_search_depth: int = 32
    max_content_scan_bytes: int = 32 * 1_024 * 1_024
    max_search_seconds: float = 5.0
    max_visible_read_bytes: int = 48 * 1_024
    max_raw_read_bytes: int = 64 * 1_024
    max_read_seconds: float = 5.0
    max_edit_source_bytes: int = 4 * 1_024 * 1_024
    max_edit_seconds: float = 10.0
    max_query_files: int = 1_000
    max_query_input_bytes: int = 256 * 1_024 * 1_024
    max_query_manifest_bytes: int = 256 * 1_024
    max_query_binding_seconds: float = 5.0

    def __post_init__(self) -> None:
        integer_limits = (
            (self.max_search_results, "max_search_results", 1, 1_000),
            (self.max_search_entries, "max_search_entries", 1, 1_000_000),
            (self.max_search_depth, "max_search_depth", 1, 128),
            (
                self.max_content_scan_bytes,
                "max_content_scan_bytes",
                1,
                1024 * 1_024 * 1_024,
            ),
            (
                self.max_visible_read_bytes,
                "max_visible_read_bytes",
                1,
                1024 * 1_024,
            ),
            (
                self.max_raw_read_bytes,
                "max_raw_read_bytes",
                4,
                2 * 1_024 * 1_024,
            ),
            (
                self.max_edit_source_bytes,
                "max_edit_source_bytes",
                1,
                64 * 1_024 * 1_024,
            ),
            (self.max_query_files, "max_query_files", 1, 1_000),
            (
                self.max_query_input_bytes,
                "max_query_input_bytes",
                1,
                256 * 1_024 * 1_024,
            ),
            (
                self.max_query_manifest_bytes,
                "max_query_manifest_bytes",
                1_024,
                256 * 1_024,
            ),
        )
        for value, name, minimum, maximum in integer_limits:
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{name} is outside its code-owned bound")
        if self.max_raw_read_bytes < self.max_visible_read_bytes + 4:
            raise ValueError("raw read window must leave room for a UTF-8 boundary")
        for seconds, name in (
            (self.max_search_seconds, "max_search_seconds"),
            (self.max_read_seconds, "max_read_seconds"),
            (self.max_edit_seconds, "max_edit_seconds"),
            (self.max_query_binding_seconds, "max_query_binding_seconds"),
        ):
            if (
                not isinstance(seconds, (int, float))
                or isinstance(seconds, bool)
                or not 0.01 <= float(seconds) <= 300.0
            ):
                raise ValueError(f"{name} is outside its code-owned bound")


@dataclass(frozen=True, slots=True)
class LocalFileBinding:
    """One transient exact physical observation authenticated by the backend."""

    workspace_id: str
    relative_path: str
    physical_revision: str
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    link_count: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    observed_at: str


@dataclass(frozen=True, slots=True)
class LocalFileQueryBinding:
    """One exact open regular file retained for a single structured query."""

    workspace_id: str
    relative_path: str
    format: str
    physical_revision: str
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    link_count: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    observed_at: str
    descriptor: int = field(repr=False, compare=False)

    def provenance_mapping(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "format": self.format,
            "physical_revision": self.physical_revision,
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "uid": self.uid,
            "gid": self.gid,
            "link_count": self.link_count,
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
            "changed_ns": self.changed_ns,
            "observed_at": self.observed_at,
        }

    def result_mapping(self) -> dict[str, object]:
        return {
            "path": self.relative_path,
            "physical_revision": self.physical_revision,
        }

    def revalidate(self) -> None:
        try:
            current = os.fstat(self.descriptor)
        except OSError as error:
            raise LocalWorkspaceError(
                "file_changed",
                "A bound workspace file became unavailable during the query.",
            ) from error
        if _physical_revision(current) != self.physical_revision:
            raise LocalWorkspaceError(
                "file_changed",
                "A bound workspace file changed during the query.",
            )


@dataclass(slots=True)
class LocalFileQueryManifest:
    """One immutable, complete, descriptor-backed input manifest."""

    workspace_id: str
    path_pattern: str
    format: str
    bindings: tuple[LocalFileQueryBinding, ...]
    input_bytes: int
    encoded_bytes: int
    manifest_sha256: str
    _closed: bool = field(default=False, init=False, repr=False, compare=False)

    def revalidate(self) -> None:
        if self._closed:
            raise LocalWorkspaceError(
                "workspace_unavailable", "The file-query manifest is closed."
            )
        for binding in self.bindings:
            binding.revalidate()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for binding in self.bindings:
            try:
                os.close(binding.descriptor)
            except OSError:
                pass

    def provenance_mapping(self) -> dict[str, object]:
        return {
            "authority": "local_workspace_binding",
            "workspace_id": self.workspace_id,
            "path_pattern": self.path_pattern,
            "format": self.format,
            "manifest_sha256": self.manifest_sha256,
            "manifest_bytes": self.encoded_bytes,
            "input_bytes": self.input_bytes,
            "bindings": tuple(item.provenance_mapping() for item in self.bindings),
        }


@dataclass(frozen=True, slots=True)
class LocalBoundFileObservation:
    """One fully observed current bound file used only by code-owned consumers."""

    binding: LocalFileBinding
    content: bytes
    content_sha256: str


@dataclass(frozen=True, slots=True)
class LocalBoundFileTarget:
    """One descriptor-contained exact target resolved for artifact delivery."""

    workspace_id: str
    relative_path: str
    filename: str
    _admitted_root_path: Path = field(repr=False, compare=False)
    parent_descriptor: int
    parent_device: int
    parent_inode: int
    parent_mode: int
    parent_uid: int
    parent_gid: int
    target_descriptor: int
    physical_revision: str
    content_sha256: str
    device: int
    inode: int
    mode: int
    uid: int
    gid: int
    link_count: int
    size_bytes: int
    modified_ns: int
    changed_ns: int
    flags: int

    def revalidate_namespace(self) -> None:
        """Prove the admitted root path still names this exact target."""

        _revalidate_bound_target_namespace(self)

    def close(self) -> None:
        os.close(self.target_descriptor)
        os.close(self.parent_descriptor)


@dataclass(frozen=True, slots=True)
class LocalFileSearchMatch:
    path: str
    match_kind: Literal["path", "content"]
    line: int | None
    excerpt: str | None
    size_bytes: int
    modified_at: str
    physical_revision: str

    def to_mapping(self) -> dict[str, object]:
        return {
            "path": self.path,
            "match_kind": self.match_kind,
            "line": self.line,
            "excerpt": self.excerpt,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
            "physical_revision": self.physical_revision,
        }


@dataclass(frozen=True, slots=True)
class LocalFileSearchResult:
    matches: tuple[LocalFileSearchMatch, ...]
    scanned_entries: int
    scanned_content_bytes: int
    truncated: bool
    truncation_reasons: tuple[str, ...]

    def to_mapping(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "matches": [item.to_mapping() for item in self.matches],
                "scanned_entries": self.scanned_entries,
                "scanned_content_bytes": self.scanned_content_bytes,
                "truncated": self.truncated,
                "truncation_reasons": self.truncation_reasons,
            }
        )


@dataclass(frozen=True, slots=True)
class LocalFileReadResult:
    path: str
    binding: str
    media_type: str
    encoding: str
    content: str
    start_offset: int
    end_offset: int
    cursor: str | None
    complete: bool
    physical_revision: str
    content_sha256: str | None
    limitations: tuple[str, ...]

    def to_mapping(self) -> FrozenJsonObject:
        return FrozenJsonObject.from_mapping(
            {
                "path": self.path,
                "binding": self.binding,
                "media_type": self.media_type,
                "encoding": self.encoding,
                "content": self.content,
                "start_offset": self.start_offset,
                "end_offset": self.end_offset,
                "cursor": self.cursor,
                "complete": self.complete,
                "physical_revision": self.physical_revision,
                "content_sha256": self.content_sha256,
                "limitations": self.limitations,
            }
        )


@dataclass(frozen=True, slots=True)
class _RootDescriptor:
    path: Path
    descriptor: int
    device: int
    inode: int


@dataclass(slots=True)
class _SearchState:
    query: str
    query_folded: str
    mode: str
    glob: str | None
    order_by: str
    limits: LocalWorkspaceLimits
    cancellation: threading.Event
    deadline: float
    matches: list[LocalFileSearchMatch]
    scanned_entries: int = 0
    scanned_content_bytes: int = 0
    total_matches: int = 0
    reasons: list[str] | None = None
    stop: bool = False

    def __post_init__(self) -> None:
        self.reasons = []

    def limit(self, reason: str, *, stop: bool = False) -> None:
        assert self.reasons is not None
        if reason not in self.reasons:
            self.reasons.append(reason)
        self.stop = self.stop or stop

    def check(self) -> None:
        if self.cancellation.is_set():
            raise _WorkerCancelled()
        if time.monotonic() >= self.deadline:
            self.limit("time_limit", stop=True)


class _WorkerCancelled(RuntimeError):
    pass


@dataclass(slots=True)
class _QueryPatternState:
    root: _RootDescriptor
    workspace_id: str
    pattern: str
    limits: LocalWorkspaceLimits
    observed_at: str
    cancellation: threading.Event
    deadline: float
    bindings: dict[str, LocalFileQueryBinding]
    scanned_entries: int = 0
    input_bytes: int = 0

    def check(self) -> None:
        _check_cancelled(self.cancellation)
        if time.monotonic() >= self.deadline:
            raise LocalWorkspaceError(
                "file_query_timeout",
                "File-query pattern binding exceeded its bounded worker time.",
            )

    def count_entry(self) -> None:
        self.scanned_entries += 1
        if self.scanned_entries > self.limits.max_search_entries:
            raise LocalWorkspaceError(
                "file_pattern_too_broad",
                "The file pattern traversed too many workspace entries.",
                {"limit": self.limits.max_search_entries},
            )


class LocalWorkspaceBackend:
    """One agent-owned descriptor, token secret, and worker lifecycle."""

    def __init__(
        self,
        *,
        workspace: LocalWorkspace,
        root: _RootDescriptor,
        limits: LocalWorkspaceLimits,
        clock: Callable[[], datetime],
        query_backend: LocalFileQueryBackend,
    ) -> None:
        self.workspace_id = _workspace_id(root)
        self.sensitivity = workspace.sensitivity
        self._root = root
        self._limits = limits
        self._clock = clock
        self._query_backend = query_backend
        self._secret = secrets.token_bytes(32)
        self._session_id = secrets.token_urlsafe(18)
        self._active: dict[asyncio.Task[object], threading.Event] = {}
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    async def open(
        cls,
        workspace: LocalWorkspace,
        *,
        agent_root: Path,
        agent_home: Path,
        limits: LocalWorkspaceLimits | None = None,
        query_limits: LocalFileQueryLimits | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> "LocalWorkspaceBackend":
        if not isinstance(workspace, LocalWorkspace):
            raise TypeError("workspace must be LocalWorkspace")
        if not isinstance(agent_root, Path) or not isinstance(agent_home, Path):
            raise TypeError("agent root and home must be pathlib.Path")
        resolved_limits = limits or LocalWorkspaceLimits()
        resolved_clock = clock or (lambda: datetime.now(UTC))
        worker = asyncio.create_task(
            asyncio.to_thread(
                _admit_root,
                workspace,
                agent_root,
                agent_home,
            )
        )
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
        return cls(
            workspace=workspace,
            root=root,
            limits=resolved_limits,
            clock=resolved_clock,
            query_backend=LocalFileQueryBackend(
                scratch_parent=agent_home / "file-query-scratch",
                limits=query_limits,
            ),
        )

    @property
    def closed(self) -> bool:
        return self._closed

    async def search(
        self,
        *,
        run_id: str,
        query: str,
        path: str = ".",
        mode: str = "paths",
        glob: str | None = None,
        order_by: str = "path",
    ) -> LocalFileSearchResult:
        _required_run_id(run_id)
        query = _search_query(query)
        relative_directory = _logical_path(path, allow_root=True)
        if mode not in {"paths", "content", "both"}:
            raise LocalWorkspaceError(
                "search_invalid", "File search mode must be paths, content, or both."
            )
        if order_by not in {"path", "modified_desc"}:
            raise LocalWorkspaceError(
                "search_invalid", "File search order must be path or modified_desc."
            )
        if glob is not None:
            if (
                not isinstance(glob, str)
                or not glob
                or len(glob) > _MAX_GLOB_CHARACTERS
                or any(character in glob for character in ("/", "\\", "\x00"))
            ):
                raise LocalWorkspaceError(
                    "search_invalid",
                    "File search glob must be one bounded filename glob.",
                )
        descriptor = self._duplicate_root()
        cancellation = threading.Event()
        return await self._run_worker(
            _search_sync,
            descriptor,
            self._root,
            relative_directory,
            query,
            mode,
            glob,
            order_by,
            self._limits,
            cancellation,
            timeout=self._limits.max_search_seconds + 1.0,
            timeout_code="search_timeout",
            timeout_message="File search exceeded its bounded worker time.",
        )

    async def read(
        self,
        *,
        run_id: str,
        path: str | None = None,
        cursor: str | None = None,
        position: str | None = None,
    ) -> LocalFileReadResult:
        _required_run_id(run_id)
        if (path is None) == (cursor is None):
            raise LocalWorkspaceError(
                "path_invalid", "File read requires exactly one path or cursor."
            )
        if position not in {None, "start", "end"}:
            raise LocalWorkspaceError(
                "path_invalid", "File read position must be start or end."
            )
        expected_revision: str | None = None
        next_offset: int | None = None
        direction = "forward"
        if cursor is not None:
            if position is not None:
                raise LocalWorkspaceError(
                    "cursor_invalid", "A cursor cannot be combined with a position."
                )
            payload = self._decode_token(cursor, purpose="cursor", run_id=run_id)
            relative_path = _logical_path(_token_text(payload, "path"))
            expected_revision = _token_text(payload, "physical_revision")
            raw_offset = payload.get("next_offset")
            raw_direction = payload.get("direction")
            if (
                not isinstance(raw_offset, int)
                or isinstance(raw_offset, bool)
                or raw_offset < 0
                or raw_direction not in {"forward", "backward"}
            ):
                raise LocalWorkspaceError(
                    "cursor_invalid", "The file cursor is malformed or unavailable."
                )
            next_offset = raw_offset
            direction = raw_direction
        else:
            assert path is not None
            relative_path = _logical_path(path)
            direction = "backward" if position == "end" else "forward"
        descriptor = self._duplicate_root()
        cancellation = threading.Event()
        observed = await self._run_worker(
            _read_sync,
            descriptor,
            self._root,
            relative_path,
            expected_revision,
            next_offset,
            direction,
            self._limits,
            cancellation,
            timeout=self._limits.max_read_seconds + 1.0,
            timeout_code="file_read_timeout",
            timeout_message="File read exceeded its bounded worker time.",
        )
        (
            content,
            start_offset,
            end_offset,
            complete,
            facts,
            content_hash,
        ) = observed
        binding = _binding_from_facts(
            workspace_id=self.workspace_id,
            relative_path=relative_path,
            facts=facts,
            observed_at=_utc_iso(self._clock()),
        )
        binding_token = self._encode_token(
            "binding",
            run_id,
            _binding_payload(binding),
        )
        next_cursor = None
        if not complete:
            next_cursor = self._encode_token(
                "cursor",
                run_id,
                {
                    "path": relative_path,
                    "physical_revision": binding.physical_revision,
                    "next_offset": (
                        end_offset if direction == "forward" else start_offset
                    ),
                    "direction": direction,
                },
            )
        return LocalFileReadResult(
            path=relative_path,
            binding=binding_token,
            media_type=_media_type(relative_path),
            encoding="utf-8",
            content=content,
            start_offset=start_offset,
            end_offset=end_offset,
            cursor=next_cursor,
            complete=complete,
            physical_revision=binding.physical_revision,
            content_sha256=content_hash,
            limitations=(() if complete else ("model_visible_byte_limit",)),
        )

    async def bind_query_manifest(
        self,
        *,
        run_id: str,
        path_pattern: str,
    ) -> LocalFileQueryManifest:
        """Expand and retain one exact workspace-relative structured dataset."""

        _required_run_id(run_id)
        pattern_parts = _query_pattern(path_pattern)
        descriptor = self._duplicate_root()
        cancellation = threading.Event()
        worker: asyncio.Task[LocalFileQueryManifest] = asyncio.create_task(
            asyncio.to_thread(
                _bind_query_manifest_sync,
                descriptor,
                self._root,
                self.workspace_id,
                path_pattern,
                pattern_parts,
                self._limits,
                _utc_iso(self._clock()),
                cancellation,
            )
        )
        self._active[worker] = cancellation
        try:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(worker),
                    timeout=self._limits.max_query_binding_seconds + 1.0,
                )
            except TimeoutError as error:
                cancellation.set()
                await _settle_worker(worker)
                _close_query_manifest_result(worker)
                raise LocalWorkspaceError(
                    "file_query_timeout",
                    "File-query pattern binding exceeded its bounded worker time.",
                ) from error
            except asyncio.CancelledError:
                cancellation.set()
                await _settle_worker(worker)
                _close_query_manifest_result(worker)
                raise
        finally:
            self._active.pop(worker, None)

    async def query(
        self,
        *,
        run_id: str,
        path_pattern: str,
        canonical_sql: str,
        sql_fingerprint: str,
    ) -> LocalFileQueryResult:
        """Bind one exact manifest and execute it through the private backend."""

        manifest = await self.bind_query_manifest(
            run_id=run_id,
            path_pattern=path_pattern,
        )
        try:
            return await self._query_backend.query(
                manifest=manifest,
                canonical_sql=canonical_sql,
                sql_fingerprint=sql_fingerprint,
            )
        finally:
            manifest.close()

    def authenticate_file_binding(
        self,
        *,
        run_id: str,
        token: str,
    ) -> LocalFileBinding:
        """Authenticate one exact binding without touching a model-authored path."""

        _required_run_id(run_id)
        payload = self._decode_token(token, purpose="binding", run_id=run_id)
        try:
            return LocalFileBinding(
                workspace_id=_token_text(payload, "workspace_id"),
                relative_path=_logical_path(_token_text(payload, "path")),
                physical_revision=_token_text(payload, "physical_revision"),
                device=_token_integer(payload, "device"),
                inode=_token_integer(payload, "inode"),
                mode=_token_integer(payload, "mode"),
                uid=_token_integer(payload, "uid"),
                gid=_token_integer(payload, "gid"),
                link_count=_token_integer(payload, "link_count"),
                size_bytes=_token_integer(payload, "size_bytes"),
                modified_ns=_token_integer(payload, "modified_ns"),
                changed_ns=_token_integer(payload, "changed_ns"),
                observed_at=_token_text(payload, "observed_at"),
            )
        except LocalWorkspaceError:
            raise
        except (TypeError, ValueError) as error:
            raise LocalWorkspaceError(
                "file_binding_invalid",
                "The file binding is malformed or unavailable.",
            ) from error

    async def observe_bound_text(
        self,
        *,
        run_id: str,
        token: str,
    ) -> LocalBoundFileObservation:
        """Fully observe one authenticated current-run binding without path input."""

        binding = self.authenticate_file_binding(run_id=run_id, token=token)
        descriptor = self._duplicate_root()
        cancellation = threading.Event()
        content, facts = await self._run_worker(
            _observe_bound_file_sync,
            descriptor,
            self._root,
            binding,
            self._limits.max_edit_source_bytes,
            cancellation,
            timeout=self._limits.max_edit_seconds + 1.0,
            timeout_code="file_edit_timeout",
            timeout_message="Bounded file observation exceeded its worker time.",
        )
        current = _binding_from_facts(
            workspace_id=self.workspace_id,
            relative_path=binding.relative_path,
            facts=facts,
            observed_at=_utc_iso(self._clock()),
        )
        return LocalBoundFileObservation(
            binding=current,
            content=content,
            content_sha256="sha256:" + sha256(content).hexdigest(),
        )

    async def resolve_bound_target(
        self,
        *,
        workspace_id: str,
        relative_path: str,
        expected_physical_revision: str | None,
        expected_content_sha256: str,
    ) -> LocalBoundFileTarget:
        """Resolve one committed binding to exact open parent/target descriptors."""

        if workspace_id != self.workspace_id:
            raise LocalWorkspaceError(
                "workspace_unavailable",
                "The bound artifact belongs to another workspace session.",
            )
        logical_path = _logical_path(relative_path)
        if not isinstance(expected_content_sha256, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", expected_content_sha256
        ):
            raise LocalWorkspaceError(
                "file_binding_invalid", "The committed file binding is invalid."
            )
        if expected_physical_revision is not None and (
            not isinstance(expected_physical_revision, str)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_physical_revision)
        ):
            raise LocalWorkspaceError(
                "file_binding_invalid", "The committed file binding is invalid."
            )
        descriptor = self._duplicate_root()
        cancellation = threading.Event()
        return await self._run_bound_target_worker(
            _resolve_bound_target_sync,
            descriptor,
            self._root,
            self.workspace_id,
            logical_path,
            expected_physical_revision,
            expected_content_sha256,
            self._limits.max_edit_source_bytes,
            cancellation,
        )

    async def close(self) -> None:
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._close_once())
        await asyncio.shield(self._close_task)

    async def _close_once(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._query_backend.close()
        for cancellation in tuple(self._active.values()):
            cancellation.set()
        for worker in tuple(self._active):
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except BaseException:
                    break
        self._active.clear()
        descriptor = self._root.descriptor
        self._root = _RootDescriptor(
            path=self._root.path,
            descriptor=-1,
            device=self._root.device,
            inode=self._root.inode,
        )
        os.close(descriptor)
        self._secret = b""

    def _duplicate_root(self) -> int:
        if self._closed or self._root.descriptor < 0:
            raise LocalWorkspaceError(
                "workspace_unavailable", "The local workspace session is closed."
            )
        try:
            return os.dup(self._root.descriptor)
        except OSError as error:
            raise LocalWorkspaceError(
                "workspace_unavailable", "The local workspace is unavailable."
            ) from error

    async def _run_worker(
        self,
        function: Callable[..., _T],
        *arguments: object,
        timeout: float,
        timeout_code: str,
        timeout_message: str,
    ) -> _T:
        cancellation = next(
            item for item in reversed(arguments) if isinstance(item, threading.Event)
        )
        worker: asyncio.Task[_T] = asyncio.create_task(
            asyncio.to_thread(function, *arguments)
        )
        self._active[worker] = cancellation
        try:
            try:
                return await asyncio.wait_for(asyncio.shield(worker), timeout=timeout)
            except TimeoutError as error:
                cancellation.set()
                await _settle_worker(worker)
                raise LocalWorkspaceError(timeout_code, timeout_message) from error
            except asyncio.CancelledError:
                cancellation.set()
                await _settle_worker(worker)
                raise
        finally:
            self._active.pop(worker, None)

    async def _run_bound_target_worker(
        self,
        function: Callable[..., LocalBoundFileTarget],
        *arguments: object,
    ) -> LocalBoundFileTarget:
        """Settle and close a descriptor-bearing result if its caller is interrupted."""

        cancellation = next(
            item for item in reversed(arguments) if isinstance(item, threading.Event)
        )
        worker = asyncio.create_task(asyncio.to_thread(function, *arguments))
        self._active[worker] = cancellation
        try:
            try:
                return await asyncio.wait_for(
                    asyncio.shield(worker),
                    timeout=self._limits.max_edit_seconds + 1.0,
                )
            except TimeoutError as error:
                cancellation.set()
                await _settle_worker(worker)
                _close_bound_target_result(worker)
                raise LocalWorkspaceError(
                    "file_edit_timeout",
                    "Bound file target resolution exceeded its worker time.",
                ) from error
            except asyncio.CancelledError:
                cancellation.set()
                await _settle_worker(worker)
                _close_bound_target_result(worker)
                raise
        finally:
            self._active.pop(worker, None)

    def _encode_token(
        self,
        purpose: str,
        run_id: str,
        values: dict[str, object],
    ) -> str:
        payload = {
            "purpose": purpose,
            "session_id": self._session_id,
            "workspace_id": self.workspace_id,
            "run_id": run_id,
            **values,
        }
        encoded = canonical_json(payload).encode("utf-8")
        signature = hmac.new(
            self._secret,
            purpose.encode("ascii") + b"\x00" + encoded,
            sha256,
        ).digest()
        token = ".".join(
            (
                _TOKEN_VERSION,
                _base64_encode(encoded),
                _base64_encode(signature),
            )
        )
        if len(token.encode("ascii")) > _MAX_TOKEN_BYTES:
            raise LocalWorkspaceError(
                "file_binding_invalid",
                "The authenticated file token exceeds its bound.",
            )
        return token

    def _decode_token(
        self,
        token: str,
        *,
        purpose: str,
        run_id: str,
    ) -> dict[str, object]:
        invalid_code = (
            "cursor_invalid" if purpose == "cursor" else "file_binding_invalid"
        )
        expired_code = (
            "cursor_expired" if purpose == "cursor" else "file_binding_expired"
        )
        invalid_message = (
            "The file cursor is malformed or unavailable."
            if purpose == "cursor"
            else "The file binding is malformed or unavailable."
        )
        expired_message = (
            "The file cursor belongs to another run or workspace session."
            if purpose == "cursor"
            else "The file binding belongs to another run or workspace session."
        )
        if (
            not isinstance(token, str)
            or not token
            or len(token.encode("utf-8")) > _MAX_TOKEN_BYTES
        ):
            raise LocalWorkspaceError(invalid_code, invalid_message)
        try:
            version, encoded, signature_text = token.split(".")
            if version != _TOKEN_VERSION:
                raise ValueError("token version")
            payload_bytes = _base64_decode(encoded)
            signature = _base64_decode(signature_text)
            if len(payload_bytes) > _MAX_TOKEN_BYTES or len(signature) != 32:
                raise ValueError("token bounds")
            loaded = json.loads(payload_bytes)
            if not isinstance(loaded, dict) or any(
                not isinstance(key, str) for key in loaded
            ):
                raise ValueError("token payload")
            payload = loaded
        except (UnicodeError, ValueError, json.JSONDecodeError) as error:
            raise LocalWorkspaceError(invalid_code, invalid_message) from error
        if payload.get("session_id") != self._session_id:
            raise LocalWorkspaceError(expired_code, expired_message)
        expected = hmac.new(
            self._secret,
            purpose.encode("ascii") + b"\x00" + payload_bytes,
            sha256,
        ).digest()
        if not hmac.compare_digest(signature, expected):
            raise LocalWorkspaceError(invalid_code, invalid_message)
        if (
            payload.get("purpose") != purpose
            or payload.get("workspace_id") != self.workspace_id
            or payload.get("run_id") != run_id
        ):
            if payload.get("run_id") != run_id:
                raise LocalWorkspaceError(expired_code, expired_message)
            raise LocalWorkspaceError(invalid_code, invalid_message)
        return payload


async def _settle_worker(worker: asyncio.Task[object]) -> None:
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            continue
        except BaseException:
            break


def _close_bound_target_result(
    worker: asyncio.Task[LocalBoundFileTarget],
) -> None:
    if worker.cancelled():
        return
    try:
        target = worker.result()
    except BaseException:
        return
    target.close()


def _close_query_manifest_result(
    worker: asyncio.Task[LocalFileQueryManifest],
) -> None:
    if worker.cancelled():
        return
    try:
        manifest = worker.result()
    except BaseException:
        return
    manifest.close()


def _admit_root(
    workspace: LocalWorkspace,
    agent_root: Path,
    agent_home: Path,
) -> _RootDescriptor:
    _require_supported_platform()
    try:
        state_root = agent_root.resolve(strict=True)
        state_home = agent_home.resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise LocalWorkspaceError(
            "workspace_state_unavailable",
            "Daita state paths could not be validated for workspace admission.",
        ) from error
    if paths_overlap(workspace.root, state_root) or paths_overlap(
        workspace.root, state_home
    ):
        raise LocalWorkspaceError(
            "workspace_state_overlap",
            "The workspace and private Daita state must not contain one another.",
        )
    return _open_root(workspace.root)


def _require_supported_platform() -> None:
    if (
        os.name != "posix"
        or not getattr(os, "O_NOFOLLOW", 0)
        or os.open not in os.supports_dir_fd
        or os.stat not in os.supports_dir_fd
    ):
        raise LocalWorkspaceError(
            "workspace_platform_unsupported",
            "This platform cannot provide secure descriptor-relative workspace access.",
        )


def _root_flags() -> int:
    return (
        os.O_RDONLY
        | os.O_DIRECTORY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _open_root(path: Path) -> _RootDescriptor:
    descriptors: list[int] = []
    bindings: list[tuple[int, str, int]] = []
    try:
        descriptor = os.open(path.anchor, _root_flags())
        descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        direct = os.stat(path.anchor, follow_symlinks=False)
        if not stat.S_ISDIR(opened.st_mode) or not _same_identity(opened, direct):
            raise OSError("unstable root anchor")
        for component in path.parts[1:]:
            direct = os.stat(component, dir_fd=descriptor, follow_symlinks=False)
            if not stat.S_ISDIR(direct.st_mode) or stat.S_ISLNK(direct.st_mode):
                raise OSError("workspace component is not a real directory")
            child = os.open(component, _root_flags(), dir_fd=descriptor)
            descriptors.append(child)
            child_opened = os.fstat(child)
            if not stat.S_ISDIR(child_opened.st_mode) or not _same_identity(
                direct, child_opened
            ):
                raise OSError("workspace component changed during admission")
            bindings.append((descriptor, component, child))
            descriptor = child
            opened = child_opened
        for parent, component, child in bindings:
            current = os.stat(component, dir_fd=parent, follow_symlinks=False)
            if not _same_identity(current, os.fstat(child)):
                raise OSError("workspace component changed during admission")
        for ancestor in descriptors[:-1]:
            os.close(ancestor)
        return _RootDescriptor(
            path=path,
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
        raise LocalWorkspaceError(
            "workspace_unavailable", "The local workspace could not be admitted safely."
        ) from error


def _verify_root_path(expected: _RootDescriptor) -> None:
    actual = _open_root(expected.path)
    try:
        if (actual.device, actual.inode) != (expected.device, expected.inode):
            raise LocalWorkspaceError(
                "workspace_identity_changed",
                "The local workspace identity changed after admission.",
            )
    finally:
        os.close(actual.descriptor)


def _workspace_id(root: _RootDescriptor) -> str:
    material = canonical_json({"device": root.device, "inode": root.inode})
    return "workspace:sha256:" + sha256(material.encode("utf-8")).hexdigest()


def _physical_revision(value: os.stat_result) -> str:
    material = {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "mode": int(value.st_mode),
        "uid": int(value.st_uid),
        "gid": int(value.st_gid),
        "link_count": int(value.st_nlink),
        "size_bytes": int(value.st_size),
        "modified_ns": int(value.st_mtime_ns),
        "changed_ns": int(value.st_ctime_ns),
        "flags": int(getattr(value, "st_flags", 0)),
    }
    return "sha256:" + sha256(canonical_json(material).encode("utf-8")).hexdigest()


def physical_revision_for_facts(value: os.stat_result) -> str:
    """Return the canonical physical revision for a code-observed regular file."""

    return _physical_revision(value)


def _bind_query_manifest_sync(
    root_descriptor: int,
    root: _RootDescriptor,
    workspace_id: str,
    pattern: str,
    pattern_parts: tuple[str, ...],
    limits: LocalWorkspaceLimits,
    observed_at: str,
    cancellation: threading.Event,
) -> LocalFileQueryManifest:
    state = _QueryPatternState(
        root=root,
        workspace_id=workspace_id,
        pattern=pattern,
        limits=limits,
        observed_at=observed_at,
        cancellation=cancellation,
        deadline=time.monotonic() + limits.max_query_binding_seconds,
        bindings={},
    )
    completed = False
    try:
        _verify_root_path(root)
        opened = os.fstat(root_descriptor)
        if (int(opened.st_dev), int(opened.st_ino)) != (root.device, root.inode):
            raise LocalWorkspaceError(
                "workspace_identity_changed",
                "The local workspace identity changed after admission.",
            )
        _expand_query_pattern(
            root_descriptor,
            (),
            pattern_parts,
            0,
            depth=0,
            state=state,
        )
        state.check()
        if not state.bindings:
            raise LocalWorkspaceError(
                "file_pattern_empty",
                "The workspace file pattern matched no supported regular files.",
            )
        bindings = tuple(
            state.bindings[path]
            for path in sorted(state.bindings, key=lambda item: item.encode("utf-8"))
        )
        formats = {item.format for item in bindings}
        if len(formats) != 1:
            raise LocalWorkspaceError(
                "format_unsupported",
                "A file query requires one homogeneous structured-file format.",
                {"formats": tuple(sorted(formats))},
            )
        format_name = next(iter(formats))
        manifest_material = {
            "protocol": "daita.local_file_query_manifest.v1",
            "workspace_id": workspace_id,
            "path_pattern": pattern,
            "format": format_name,
            "bindings": [item.provenance_mapping() for item in bindings],
        }
        encoded = canonical_json(manifest_material).encode("utf-8")
        if len(encoded) > limits.max_query_manifest_bytes:
            raise LocalWorkspaceError(
                "file_pattern_too_broad",
                "The complete file-query binding manifest exceeds its byte bound.",
                {
                    "limit": limits.max_query_manifest_bytes,
                    "observed": len(encoded),
                },
            )
        for binding in bindings:
            binding.revalidate()
        _verify_root_path(root)
        result = LocalFileQueryManifest(
            workspace_id=workspace_id,
            path_pattern=pattern,
            format=format_name,
            bindings=bindings,
            input_bytes=state.input_bytes,
            encoded_bytes=len(encoded),
            manifest_sha256="sha256:" + sha256(encoded).hexdigest(),
        )
        completed = True
        return result
    except _WorkerCancelled:
        raise
    except LocalWorkspaceError:
        raise
    except OSError as error:
        raise LocalWorkspaceError(
            "workspace_unavailable",
            "The workspace file pattern could not be bound safely.",
        ) from error
    finally:
        os.close(root_descriptor)
        if not completed:
            for binding in state.bindings.values():
                try:
                    os.close(binding.descriptor)
                except OSError:
                    pass


def _expand_query_pattern(
    directory_descriptor: int,
    parents: tuple[str, ...],
    pattern_parts: tuple[str, ...],
    index: int,
    *,
    depth: int,
    state: _QueryPatternState,
) -> None:
    state.check()
    if index >= len(pattern_parts):
        return
    segment = pattern_parts[index]
    if segment == "**":
        _expand_query_pattern(
            directory_descriptor,
            parents,
            pattern_parts,
            index + 1,
            depth=depth,
            state=state,
        )
        if depth >= state.limits.max_search_depth:
            raise LocalWorkspaceError(
                "file_pattern_too_broad",
                "The file pattern exceeded the workspace recursion bound.",
                {"limit": state.limits.max_search_depth},
            )
        for name in _query_directory_names(directory_descriptor, state):
            if _restricted_component(name):
                continue
            try:
                before = os.stat(
                    name,
                    dir_fd=directory_descriptor,
                    follow_symlinks=False,
                )
            except OSError as error:
                raise LocalWorkspaceError(
                    "workspace_unavailable",
                    "The workspace changed during file-pattern expansion.",
                ) from error
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                continue
            child = _open_checked_directory(directory_descriptor, name, before)
            try:
                _expand_query_pattern(
                    child,
                    (*parents, name),
                    pattern_parts,
                    index,
                    depth=depth + 1,
                    state=state,
                )
            finally:
                os.close(child)
        return

    last = index == len(pattern_parts) - 1
    for name in _query_directory_names(directory_descriptor, state):
        if not fnmatch.fnmatchcase(name, segment):
            continue
        if _restricted_component(name):
            continue
        try:
            before = os.stat(
                name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except OSError as error:
            raise LocalWorkspaceError(
                "workspace_unavailable",
                "The workspace changed during file-pattern expansion.",
            ) from error
        if stat.S_ISLNK(before.st_mode):
            raise LocalWorkspaceError(
                "symlink_not_allowed",
                "Workspace file patterns cannot traverse or select symlinks.",
            )
        if last:
            if not stat.S_ISREG(before.st_mode):
                if not _pattern_has_magic(segment):
                    raise LocalWorkspaceError(
                        "not_regular_file",
                        "The workspace file pattern selected a non-regular file.",
                    )
                continue
            _bind_query_file(directory_descriptor, parents, name, before, state)
            continue
        if not stat.S_ISDIR(before.st_mode):
            if not _pattern_has_magic(segment):
                raise LocalWorkspaceError(
                    "path_invalid",
                    "A workspace file-pattern component is not a directory.",
                )
            continue
        if depth >= state.limits.max_search_depth:
            raise LocalWorkspaceError(
                "file_pattern_too_broad",
                "The file pattern exceeded the workspace recursion bound.",
                {"limit": state.limits.max_search_depth},
            )
        child = _open_checked_directory(directory_descriptor, name, before)
        try:
            _expand_query_pattern(
                child,
                (*parents, name),
                pattern_parts,
                index + 1,
                depth=depth + 1,
                state=state,
            )
        finally:
            os.close(child)


def _query_directory_names(
    directory_descriptor: int,
    state: _QueryPatternState,
) -> tuple[str, ...]:
    state.check()
    try:
        names = tuple(
            sorted(
                (
                    name
                    for name in os.listdir(directory_descriptor)
                    if _safe_segment(name)
                ),
                key=lambda name: name.encode("utf-8"),
            )
        )
    except OSError as error:
        raise LocalWorkspaceError(
            "workspace_unavailable",
            "A workspace directory became unavailable during file-pattern expansion.",
        ) from error
    for _name in names:
        state.count_entry()
    return names


def _bind_query_file(
    directory_descriptor: int,
    parents: tuple[str, ...],
    name: str,
    before: os.stat_result,
    state: _QueryPatternState,
) -> None:
    relative_path = PurePosixPath(*parents, name).as_posix()
    if relative_path in state.bindings:
        return
    format_name = _LOCAL_QUERY_FORMATS.get(PurePosixPath(name).suffix.casefold())
    if format_name is None:
        raise LocalWorkspaceError(
            "format_unsupported",
            "The file pattern matched an unsupported structured-file format.",
            {"path": relative_path},
        )
    if len(state.bindings) >= state.limits.max_query_files:
        raise LocalWorkspaceError(
            "file_pattern_too_broad",
            "The file pattern matched too many files.",
            {"limit": state.limits.max_query_files},
        )
    descriptor = _open_checked_file(directory_descriptor, name, before)
    try:
        facts = os.fstat(descriptor)
        size_bytes = int(facts.st_size)
        if state.input_bytes + size_bytes > state.limits.max_query_input_bytes:
            raise LocalWorkspaceError(
                "file_pattern_too_broad",
                "The file pattern exceeds the total physical input-byte bound.",
                {
                    "limit": state.limits.max_query_input_bytes,
                    "observed": state.input_bytes + size_bytes,
                },
            )
        binding = LocalFileQueryBinding(
            workspace_id=state.workspace_id,
            relative_path=relative_path,
            format=format_name,
            physical_revision=_physical_revision(facts),
            device=int(facts.st_dev),
            inode=int(facts.st_ino),
            mode=int(facts.st_mode),
            uid=int(facts.st_uid),
            gid=int(facts.st_gid),
            link_count=int(facts.st_nlink),
            size_bytes=size_bytes,
            modified_ns=int(facts.st_mtime_ns),
            changed_ns=int(facts.st_ctime_ns),
            observed_at=state.observed_at,
            descriptor=descriptor,
        )
        state.bindings[relative_path] = binding
        state.input_bytes += size_bytes
        descriptor = -1
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _search_sync(
    root_descriptor: int,
    root: _RootDescriptor,
    relative_directory: str,
    query: str,
    mode: str,
    glob: str | None,
    order_by: str,
    limits: LocalWorkspaceLimits,
    cancellation: threading.Event,
) -> LocalFileSearchResult:
    state = _SearchState(
        query=query,
        query_folded=query.casefold(),
        mode=mode,
        glob=glob,
        order_by=order_by,
        limits=limits,
        cancellation=cancellation,
        deadline=time.monotonic() + limits.max_search_seconds,
        matches=[],
    )
    directory_descriptor = root_descriptor
    try:
        _verify_root_path(root)
        root_opened = os.fstat(root_descriptor)
        if (int(root_opened.st_dev), int(root_opened.st_ino)) != (
            root.device,
            root.inode,
        ):
            raise LocalWorkspaceError(
                "workspace_identity_changed",
                "The local workspace identity changed after admission.",
            )
        parents = (
            () if relative_directory == "." else PurePosixPath(relative_directory).parts
        )
        for component in parents:
            _deny_restricted_component(component)
            before = os.stat(
                component, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if stat.S_ISLNK(before.st_mode):
                raise LocalWorkspaceError(
                    "symlink_not_allowed", "Workspace paths cannot traverse symlinks."
                )
            if not stat.S_ISDIR(before.st_mode):
                raise LocalWorkspaceError(
                    "not_directory", "The file-search path is not a directory."
                )
            child = _open_checked_directory(directory_descriptor, component, before)
            if directory_descriptor != root_descriptor:
                os.close(directory_descriptor)
            directory_descriptor = child
        _scan_directory(directory_descriptor, parents, depth=0, state=state)
        assert state.reasons is not None
        if order_by == "modified_desc":
            state.matches.sort(
                key=lambda item: (
                    -_modified_sort_value(item.modified_at),
                    item.path.encode("utf-8"),
                    item.line or 0,
                    item.match_kind,
                )
            )
        else:
            state.matches.sort(
                key=lambda item: (
                    item.path.encode("utf-8"),
                    item.line or 0,
                    item.match_kind,
                )
            )
        selected = tuple(state.matches[: limits.max_search_results])
        if state.total_matches > len(selected):
            state.limit("result_limit")
        return LocalFileSearchResult(
            matches=selected,
            scanned_entries=state.scanned_entries,
            scanned_content_bytes=state.scanned_content_bytes,
            truncated=bool(state.reasons),
            truncation_reasons=tuple(state.reasons),
        )
    except _WorkerCancelled:
        raise
    except LocalWorkspaceError:
        raise
    except OSError as error:
        raise LocalWorkspaceError(
            "workspace_unavailable", "File search could not complete safely."
        ) from error
    finally:
        if directory_descriptor != root_descriptor:
            os.close(directory_descriptor)
        os.close(root_descriptor)


def _scan_directory(
    directory_descriptor: int,
    parents: tuple[str, ...],
    *,
    depth: int,
    state: _SearchState,
) -> None:
    state.check()
    if state.stop:
        return
    try:
        names = tuple(
            sorted(
                (
                    name
                    for name in os.listdir(directory_descriptor)
                    if _safe_segment(name)
                ),
                key=lambda name: name.encode("utf-8"),
            )
        )
    except OSError:
        state.limit("entry_unavailable")
        return
    for name in names:
        state.check()
        if state.stop:
            return
        if state.scanned_entries >= state.limits.max_search_entries:
            state.limit("entry_limit", stop=True)
            return
        state.scanned_entries += 1
        if _restricted_component(name):
            continue
        try:
            before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        except OSError:
            state.limit("entry_unavailable")
            continue
        if stat.S_ISLNK(before.st_mode):
            continue
        if stat.S_ISDIR(before.st_mode):
            if name.casefold() in _SEARCH_NOISE_DIRECTORIES:
                continue
            if depth >= state.limits.max_search_depth:
                state.limit("depth_limit")
                continue
            try:
                child = _open_checked_directory(directory_descriptor, name, before)
            except OSError:
                state.limit("entry_unavailable")
                continue
            try:
                _scan_directory(child, (*parents, name), depth=depth + 1, state=state)
            finally:
                os.close(child)
            continue
        if not stat.S_ISREG(before.st_mode):
            continue
        if state.glob is not None and not fnmatch.fnmatchcase(name, state.glob):
            continue
        relative_path = PurePosixPath(*parents, name).as_posix()
        path_matches = state.query_folded in relative_path.casefold()
        if state.mode in {"paths", "both"} and path_matches:
            _add_search_match(
                state,
                _search_match(relative_path, "path", before, line=None, excerpt=None),
            )
        if state.mode not in {"content", "both"}:
            continue
        remaining = state.limits.max_content_scan_bytes - state.scanned_content_bytes
        if remaining <= 0:
            state.limit("content_byte_limit", stop=True)
            return
        try:
            content_matches, limited = _search_file_content(
                directory_descriptor,
                name,
                before,
                state.query_folded,
                min(remaining, state.limits.max_content_scan_bytes),
                state,
            )
        except OSError:
            if state.scanned_content_bytes >= state.limits.max_content_scan_bytes:
                state.limit("content_byte_limit", stop=True)
                return
            continue
        if limited:
            state.limit("content_byte_limit", stop=True)
        for line, excerpt in content_matches:
            _add_search_match(
                state,
                _search_match(
                    relative_path,
                    "content",
                    before,
                    line=line,
                    excerpt=excerpt,
                ),
            )
        if state.stop:
            return


def _add_search_match(state: _SearchState, match: LocalFileSearchMatch) -> None:
    state.total_matches += 1
    if (
        state.order_by == "path"
        and len(state.matches) >= state.limits.max_search_results
    ):
        state.limit("result_limit", stop=True)
        return
    state.matches.append(match)
    if (
        state.order_by == "modified_desc"
        and len(state.matches) > state.limits.max_search_results * 2
    ):
        state.matches.sort(
            key=lambda item: (
                -_modified_sort_value(item.modified_at),
                item.path.encode("utf-8"),
                item.line or 0,
                item.match_kind,
            )
        )
        del state.matches[state.limits.max_search_results :]


def _search_file_content(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
    query_folded: str,
    maximum_bytes: int,
    state: _SearchState,
) -> tuple[list[tuple[int, str]], bool]:
    descriptor = _open_checked_file(parent_descriptor, name, before)
    decoder = codecs.getincrementaldecoder("utf-8")("strict")
    matches: list[tuple[int, str]] = []
    buffer = ""
    base_line = 1
    scanned = 0
    limited = False
    try:
        while scanned < maximum_bytes:
            state.check()
            if state.stop:
                break
            chunk = os.read(descriptor, min(_READ_CHUNK_BYTES, maximum_bytes - scanned))
            if not chunk:
                try:
                    buffer += decoder.decode(b"", final=True)
                except UnicodeError:
                    return [], False
                break
            scanned += len(chunk)
            state.scanned_content_bytes += len(chunk)
            if b"\x00" in chunk:
                return [], False
            try:
                buffer += decoder.decode(chunk, final=False)
            except UnicodeError:
                limited = (
                    scanned >= maximum_bytes and os.fstat(descriptor).st_size > scanned
                )
                return [], limited
            base_line = _collect_content_matches(
                buffer,
                query_folded,
                base_line,
                matches,
            )[0]
            keep = max(len(query_folded) + _MAX_EXCERPT_CHARACTERS, 512)
            if len(buffer) > keep:
                dropped = buffer[:-keep]
                base_line += dropped.count("\n")
                buffer = buffer[-keep:]
            if len(matches) >= state.limits.max_search_results:
                break
        if scanned >= maximum_bytes and os.fstat(descriptor).st_size > scanned:
            limited = True
        if buffer and len(matches) < state.limits.max_search_results:
            _collect_content_matches(buffer, query_folded, base_line, matches)
        finished = os.fstat(descriptor)
        if not _same_file_version(before, finished):
            raise OSError("file changed during content search")
        deduplicated = list(dict.fromkeys(matches))
        return deduplicated[: state.limits.max_search_results], limited
    finally:
        os.close(descriptor)


def _collect_content_matches(
    text: str,
    query_folded: str,
    base_line: int,
    matches: list[tuple[int, str]],
) -> tuple[int, int]:
    folded = text.casefold()
    start = 0
    while len(matches) < 100:
        index = folded.find(query_folded, start)
        if index < 0:
            break
        line = base_line + text[:index].count("\n")
        line_start = text.rfind("\n", 0, index) + 1
        line_end = text.find("\n", index)
        if line_end < 0:
            line_end = len(text)
        excerpt = text[line_start:line_end]
        if len(excerpt) > _MAX_EXCERPT_CHARACTERS:
            relative = index - line_start
            left = max(0, relative - _MAX_EXCERPT_CHARACTERS // 2)
            excerpt = excerpt[left : left + _MAX_EXCERPT_CHARACTERS]
        matches.append((line, excerpt))
        start = index + max(1, len(query_folded))
    return base_line, start


def _search_match(
    path: str,
    kind: Literal["path", "content"],
    facts: os.stat_result,
    *,
    line: int | None,
    excerpt: str | None,
) -> LocalFileSearchMatch:
    return LocalFileSearchMatch(
        path=path,
        match_kind=kind,
        line=line,
        excerpt=excerpt,
        size_bytes=int(facts.st_size),
        modified_at=_timestamp_ns(int(facts.st_mtime_ns)),
        physical_revision=_physical_revision(facts),
    )


def _read_sync(
    root_descriptor: int,
    root: _RootDescriptor,
    relative_path: str,
    expected_revision: str | None,
    next_offset: int | None,
    direction: str,
    limits: LocalWorkspaceLimits,
    cancellation: threading.Event,
) -> tuple[str, int, int, bool, os.stat_result, str | None]:
    descriptor = -1
    try:
        _verify_root_path(root)
        _check_cancelled(cancellation)
        descriptor, before = _open_relative_file(root_descriptor, relative_path)
        revision = _physical_revision(before)
        if expected_revision is not None and revision != expected_revision:
            raise LocalWorkspaceError(
                "file_changed", "The file changed after the previous read chunk."
            )
        size = int(before.st_size)
        if direction == "forward":
            start = 0 if next_offset is None else next_offset
            if start > size:
                raise LocalWorkspaceError(
                    "file_changed", "The file changed after the previous read chunk."
                )
            raw = _pread(
                descriptor,
                min(limits.max_raw_read_bytes, size - start),
                start,
            )
            content_bytes = _utf8_forward(raw, limits.max_visible_read_bytes)
            end = start + len(content_bytes)
            complete = end == size
        else:
            end = size if next_offset is None else next_offset
            if end > size:
                raise LocalWorkspaceError(
                    "file_changed", "The file changed after the previous read chunk."
                )
            raw_start = max(0, end - limits.max_raw_read_bytes)
            raw = _pread(descriptor, end - raw_start, raw_start)
            content_bytes, trim = _utf8_backward(
                raw,
                limits.max_visible_read_bytes,
            )
            start = raw_start + trim
            complete = start == 0
        _check_cancelled(cancellation)
        finished = os.fstat(descriptor)
        if not _same_file_version(before, finished):
            raise LocalWorkspaceError(
                "file_changed", "The file changed during the bounded read."
            )
        content_hash = (
            "sha256:" + sha256(content_bytes).hexdigest()
            if start == 0 and end == size
            else None
        )
        return (
            content_bytes.decode("utf-8", errors="strict"),
            start,
            end,
            complete,
            finished,
            content_hash,
        )
    except LocalWorkspaceError:
        raise
    except UnicodeError as error:
        raise LocalWorkspaceError(
            "encoding_unsupported", "The file is not valid bounded UTF-8 text."
        ) from error
    except OSError as error:
        raise LocalWorkspaceError(
            "file_not_found", "The requested workspace file is unavailable."
        ) from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_descriptor)


def _observe_bound_file_sync(
    root_descriptor: int,
    root: _RootDescriptor,
    binding: LocalFileBinding,
    maximum_bytes: int,
    cancellation: threading.Event,
) -> tuple[bytes, os.stat_result]:
    descriptor = -1
    try:
        _verify_root_path(root)
        _check_cancelled(cancellation)
        descriptor, before = _open_relative_file(root_descriptor, binding.relative_path)
        if _physical_revision(before) != binding.physical_revision:
            raise LocalWorkspaceError(
                "file_changed", "The bound workspace file changed before editing."
            )
        content = _read_all_bounded(
            descriptor,
            before,
            maximum_bytes=maximum_bytes,
            cancellation=cancellation,
        )
        finished = os.fstat(descriptor)
        if not _same_file_version(before, finished):
            raise LocalWorkspaceError(
                "file_changed", "The bound workspace file changed during editing."
            )
        return content, finished
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(root_descriptor)


def _resolve_bound_target_sync(
    root_descriptor: int,
    root: _RootDescriptor,
    workspace_id: str,
    relative_path: str,
    expected_physical_revision: str | None,
    expected_content_sha256: str,
    maximum_bytes: int,
    cancellation: threading.Event,
) -> LocalBoundFileTarget:
    parent_descriptor = -1
    target_descriptor = -1
    try:
        _verify_root_path(root)
        _check_cancelled(cancellation)
        (
            parent_descriptor,
            target_descriptor,
            before,
            filename,
        ) = _open_relative_target(root_descriptor, relative_path)
        root_descriptor = -1
        revision = _physical_revision(before)
        if (
            expected_physical_revision is not None
            and revision != expected_physical_revision
        ):
            raise LocalWorkspaceError(
                "file_changed", "The bound workspace file changed before publication."
            )
        content = _read_all_bounded(
            target_descriptor,
            before,
            maximum_bytes=maximum_bytes,
            cancellation=cancellation,
        )
        finished = os.fstat(target_descriptor)
        if not _same_file_version(before, finished):
            raise LocalWorkspaceError(
                "file_changed", "The bound workspace file changed during validation."
            )
        observed_hash = "sha256:" + sha256(content).hexdigest()
        if observed_hash != expected_content_sha256:
            raise LocalWorkspaceError(
                "file_changed", "The bound workspace file content changed."
            )
        parent_facts = os.fstat(parent_descriptor)
        result = LocalBoundFileTarget(
            workspace_id=workspace_id,
            relative_path=relative_path,
            filename=filename,
            _admitted_root_path=root.path,
            parent_descriptor=parent_descriptor,
            parent_device=int(parent_facts.st_dev),
            parent_inode=int(parent_facts.st_ino),
            parent_mode=int(parent_facts.st_mode),
            parent_uid=int(parent_facts.st_uid),
            parent_gid=int(parent_facts.st_gid),
            target_descriptor=target_descriptor,
            physical_revision=revision,
            content_sha256=observed_hash,
            device=int(finished.st_dev),
            inode=int(finished.st_ino),
            mode=int(finished.st_mode),
            uid=int(finished.st_uid),
            gid=int(finished.st_gid),
            link_count=int(finished.st_nlink),
            size_bytes=int(finished.st_size),
            modified_ns=int(finished.st_mtime_ns),
            changed_ns=int(finished.st_ctime_ns),
            flags=int(getattr(finished, "st_flags", 0)),
        )
        parent_descriptor = -1
        target_descriptor = -1
        return result
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)


def _revalidate_bound_target_namespace(target: LocalBoundFileTarget) -> None:
    """Rewalk the admitted namespace and match its root, parent, and target."""

    root_descriptor = -1
    parent_descriptor = -1
    target_descriptor = -1
    try:
        actual_root = _open_root(target._admitted_root_path)
        root_descriptor = actual_root.descriptor
        if _workspace_id(actual_root) != target.workspace_id:
            raise LocalWorkspaceError(
                "workspace_identity_changed",
                "The local workspace identity changed after admission.",
            )
        consumed_root = root_descriptor
        root_descriptor = -1
        (
            parent_descriptor,
            target_descriptor,
            target_facts,
            filename,
        ) = _open_relative_target(consumed_root, target.relative_path)
        parent_facts = os.fstat(parent_descriptor)
        if (
            filename != target.filename
            or int(parent_facts.st_dev) != target.parent_device
            or int(parent_facts.st_ino) != target.parent_inode
            or int(target_facts.st_dev) != target.device
            or int(target_facts.st_ino) != target.inode
            or _physical_revision(target_facts) != target.physical_revision
        ):
            raise LocalWorkspaceError(
                "file_changed",
                "The bound workspace file changed before publication.",
            )
    except LocalWorkspaceError as error:
        raise LocalWorkspaceError(
            "file_changed",
            "The bound workspace namespace changed before publication.",
        ) from error
    except OSError as error:
        raise LocalWorkspaceError(
            "file_changed",
            "The bound workspace namespace changed before publication.",
        ) from error
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
        if parent_descriptor >= 0:
            os.close(parent_descriptor)
        if root_descriptor >= 0:
            os.close(root_descriptor)


def _read_all_bounded(
    descriptor: int,
    facts: os.stat_result,
    *,
    maximum_bytes: int,
    cancellation: threading.Event,
) -> bytes:
    size = int(facts.st_size)
    if size > maximum_bytes:
        raise LocalWorkspaceError(
            "file_too_large",
            "The workspace file exceeds the bounded text-edit size.",
            {"limit": maximum_bytes, "observed": size},
        )
    content = bytearray()
    offset = 0
    while offset < size:
        _check_cancelled(cancellation)
        chunk = os.pread(descriptor, min(_READ_CHUNK_BYTES, size - offset), offset)
        if not chunk:
            raise LocalWorkspaceError(
                "file_changed", "The workspace file changed during full observation."
            )
        content.extend(chunk)
        offset += len(chunk)
    _check_cancelled(cancellation)
    if b"\x00" in content:
        raise LocalWorkspaceError("file_binary", "The file appears to be binary.")
    try:
        bytes(content).decode("utf-8", errors="strict")
    except UnicodeError as error:
        raise LocalWorkspaceError(
            "encoding_unsupported", "The file is not valid UTF-8 text."
        ) from error
    return bytes(content)


def _open_relative_file(
    root_descriptor: int,
    relative_path: str,
) -> tuple[int, os.stat_result]:
    directory_descriptor = root_descriptor
    try:
        parts = PurePosixPath(relative_path).parts
        for component in parts[:-1]:
            _deny_restricted_component(component)
            before = os.stat(
                component, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if stat.S_ISLNK(before.st_mode):
                raise LocalWorkspaceError(
                    "symlink_not_allowed", "Workspace paths cannot traverse symlinks."
                )
            if not stat.S_ISDIR(before.st_mode):
                raise LocalWorkspaceError(
                    "path_invalid", "A workspace path component is not a directory."
                )
            child = _open_checked_directory(directory_descriptor, component, before)
            if directory_descriptor != root_descriptor:
                os.close(directory_descriptor)
            directory_descriptor = child
        name = parts[-1]
        _deny_restricted_component(name)
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise LocalWorkspaceError(
                "symlink_not_allowed", "Workspace files cannot be symlinks."
            )
        if not stat.S_ISREG(before.st_mode):
            raise LocalWorkspaceError(
                "not_regular_file",
                "The requested workspace path is not a regular file.",
            )
        descriptor = _open_checked_file(directory_descriptor, name, before)
        return descriptor, os.fstat(descriptor)
    finally:
        if directory_descriptor != root_descriptor:
            os.close(directory_descriptor)


def _open_relative_target(
    root_descriptor: int,
    relative_path: str,
) -> tuple[int, int, os.stat_result, str]:
    """Consume one duplicated root and return an exact parent and target."""

    directory_descriptor = root_descriptor
    target_descriptor = -1
    try:
        parts = PurePosixPath(relative_path).parts
        for component in parts[:-1]:
            _deny_restricted_component(component)
            before = os.stat(
                component, dir_fd=directory_descriptor, follow_symlinks=False
            )
            if stat.S_ISLNK(before.st_mode):
                raise LocalWorkspaceError(
                    "symlink_not_allowed", "Workspace paths cannot traverse symlinks."
                )
            if not stat.S_ISDIR(before.st_mode):
                raise LocalWorkspaceError(
                    "path_invalid", "A workspace path component is not a directory."
                )
            child = _open_checked_directory(directory_descriptor, component, before)
            os.close(directory_descriptor)
            directory_descriptor = child
        name = parts[-1]
        _deny_restricted_component(name)
        before = os.stat(name, dir_fd=directory_descriptor, follow_symlinks=False)
        if stat.S_ISLNK(before.st_mode):
            raise LocalWorkspaceError(
                "symlink_not_allowed", "Workspace files cannot be symlinks."
            )
        if not stat.S_ISREG(before.st_mode):
            raise LocalWorkspaceError(
                "not_regular_file",
                "The requested workspace path is not a regular file.",
            )
        target_descriptor = _open_checked_file(directory_descriptor, name, before)
        facts = os.fstat(target_descriptor)
        result = (directory_descriptor, target_descriptor, facts, name)
        directory_descriptor = -1
        target_descriptor = -1
        return result
    finally:
        if target_descriptor >= 0:
            os.close(target_descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)


def _open_checked_directory(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
) -> int:
    descriptor = os.open(name, _root_flags(), dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or not _same_identity(before, opened):
            raise OSError("directory changed during traversal")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_checked_file(
    parent_descriptor: int,
    name: str,
    before: os.stat_result,
) -> int:
    flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    descriptor = os.open(name, flags, dir_fd=parent_descriptor)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or not _same_identity(before, opened):
            raise OSError("file changed before open")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _utf8_forward(raw: bytes, maximum: int) -> bytes:
    candidate = raw[:maximum]
    if b"\x00" in candidate:
        raise LocalWorkspaceError("file_binary", "The file appears to be binary.")
    for _ in range(4):
        try:
            candidate.decode("utf-8", errors="strict")
            return candidate
        except UnicodeDecodeError as error:
            if error.end == len(candidate) and error.reason == "unexpected end of data":
                candidate = candidate[:-1]
                continue
            raise
    raise UnicodeDecodeError("utf-8", candidate, 0, len(candidate), "invalid boundary")


def _utf8_backward(raw: bytes, maximum: int) -> tuple[bytes, int]:
    candidate_start = max(0, len(raw) - maximum)
    for _ in range(4):
        candidate = raw[candidate_start:]
        if b"\x00" in candidate:
            raise LocalWorkspaceError("file_binary", "The file appears to be binary.")
        try:
            candidate.decode("utf-8", errors="strict")
            return candidate, candidate_start
        except UnicodeDecodeError as error:
            if error.start == 0 and candidate_start < len(raw):
                candidate_start += 1
                continue
            raise
    candidate = raw[candidate_start:]
    raise UnicodeDecodeError("utf-8", candidate, 0, len(candidate), "invalid boundary")


def _pread(descriptor: int, size: int, offset: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    current = offset
    while remaining:
        chunk = os.pread(descriptor, min(_READ_CHUNK_BYTES, remaining), current)
        if not chunk:
            break
        chunks.append(chunk)
        current += len(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


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


def _logical_path(value: str, *, allow_root: bool = False) -> str:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_LOGICAL_PATH_CHARACTERS
        or "\x00" in value
        or "\\" in value
    ):
        raise LocalWorkspaceError("path_invalid", "Workspace path is invalid.")
    if allow_root and value == ".":
        return value
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(not _safe_segment(component) for component in path.parts)
    ):
        raise LocalWorkspaceError("path_invalid", "Workspace path is invalid.")
    for component in path.parts:
        _deny_restricted_component(component)
    return value


def _query_pattern(value: str) -> tuple[str, ...]:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > _MAX_QUERY_PATTERN_CHARACTERS
        or "\x00" in value
        or "\\" in value
    ):
        raise LocalWorkspaceError(
            "path_invalid", "The workspace file pattern is invalid."
        )
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or not path.parts
        or any(not _safe_pattern_segment(item) for item in path.parts)
        or sum(item == "**" for item in path.parts) > 1
    ):
        raise LocalWorkspaceError(
            "path_invalid", "The workspace file pattern is invalid."
        )
    for component in path.parts:
        if not _pattern_has_magic(component):
            _deny_restricted_component(component)
    return tuple(path.parts)


def _safe_pattern_segment(value: object) -> bool:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    if any(character in value for character in ("/", "\\", "\x00")):
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    return True


def _pattern_has_magic(value: str) -> bool:
    return any(character in value for character in ("*", "?", "["))


def _safe_segment(value: object) -> bool:
    if not isinstance(value, str) or not value or value in {".", ".."}:
        return False
    if any(character in value for character in ("/", "\\", "\x00")):
        return False
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        return False
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeError:
        return False
    return True


def _restricted_component(value: str) -> bool:
    folded = value.casefold()
    suffix = PurePosixPath(folded).suffix
    return (
        folded == ".env"
        or folded.startswith(".env.")
        or folded in _RESTRICTED_DIRECTORY_NAMES
        or folded in _RESTRICTED_FILE_NAMES
        or suffix in _RESTRICTED_FILE_SUFFIXES
    )


def _deny_restricted_component(value: str) -> None:
    if _restricted_component(value):
        raise LocalWorkspaceError(
            "path_restricted", "The requested workspace path is security-restricted."
        )


def _search_query(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > _MAX_QUERY_CHARACTERS
        or "\x00" in value
    ):
        raise LocalWorkspaceError(
            "search_invalid", "File search query must be bounded literal text."
        )
    return value


def _required_run_id(value: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ValueError("run_id must be bounded non-empty text")


def _check_cancelled(cancellation: threading.Event) -> None:
    if cancellation.is_set():
        raise _WorkerCancelled()


def _binding_payload(binding: LocalFileBinding) -> dict[str, object]:
    return {
        "path": binding.relative_path,
        "physical_revision": binding.physical_revision,
        "device": binding.device,
        "inode": binding.inode,
        "mode": binding.mode,
        "uid": binding.uid,
        "gid": binding.gid,
        "link_count": binding.link_count,
        "size_bytes": binding.size_bytes,
        "modified_ns": binding.modified_ns,
        "changed_ns": binding.changed_ns,
        "observed_at": binding.observed_at,
    }


def _binding_from_facts(
    *,
    workspace_id: str,
    relative_path: str,
    facts: os.stat_result,
    observed_at: str,
) -> LocalFileBinding:
    return LocalFileBinding(
        workspace_id=workspace_id,
        relative_path=relative_path,
        physical_revision=_physical_revision(facts),
        device=int(facts.st_dev),
        inode=int(facts.st_ino),
        mode=int(facts.st_mode),
        uid=int(facts.st_uid),
        gid=int(facts.st_gid),
        link_count=int(facts.st_nlink),
        size_bytes=int(facts.st_size),
        modified_ns=int(facts.st_mtime_ns),
        changed_ns=int(facts.st_ctime_ns),
        observed_at=observed_at,
    )


def _token_text(payload: dict[str, object], name: str) -> str:
    value = payload.get(name)
    if not isinstance(value, str) or not value or len(value) > _MAX_TOKEN_BYTES:
        raise LocalWorkspaceError(
            "file_binding_invalid", "The authenticated file token is malformed."
        )
    return value


def _token_integer(payload: dict[str, object], name: str) -> int:
    value = payload.get(name)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise LocalWorkspaceError(
            "file_binding_invalid", "The authenticated file token is malformed."
        )
    return value


def _base64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _base64_decode(value: str) -> bytes:
    padded = value + "=" * (-len(value) % 4)
    decoded = base64.b64decode(padded, altchars=b"-_", validate=True)
    if _base64_encode(decoded) != value:
        raise ValueError("non-canonical base64")
    return decoded


def _timestamp_ns(value: int) -> str:
    return _utc_iso(datetime.fromtimestamp(value / 1_000_000_000, tz=UTC))


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("workspace clock must return an aware datetime")
    return (
        value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def _modified_sort_value(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _media_type(relative_path: str) -> str:
    guessed, _encoding = mimetypes.guess_type(relative_path, strict=False)
    if guessed is None or not (
        guessed.startswith("text/")
        or guessed
        in {
            "application/json",
            "application/javascript",
            "application/toml",
            "application/xml",
            "application/yaml",
        }
    ):
        return "text/plain"
    return guessed


__all__ = [
    "LocalBoundFileObservation",
    "LocalBoundFileTarget",
    "LocalFileBinding",
    "LocalFileQueryBinding",
    "LocalFileQueryManifest",
    "LocalFileReadResult",
    "LocalFileSearchMatch",
    "LocalFileSearchResult",
    "LocalWorkspaceBackend",
    "LocalWorkspaceError",
    "LocalWorkspaceLimits",
    "physical_revision_for_facts",
]
