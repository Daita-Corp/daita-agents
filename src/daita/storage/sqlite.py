"""Small SQLite state store for one embedded agent.

The database persists only product state the MVP actually uses: identity,
attached sources, current catalog snapshots, and exact run transcripts.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import threading
from typing import Any, TypeVar

from .._json import FrozenJsonObject
from ..adapters.models import SourceRegistration
from ..catalog.models import (
    CatalogFacet,
    CatalogPath,
    CatalogPathStep,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSearchHit,
    CatalogSearchRequest,
    CatalogSearchResult,
    CatalogSync,
    CatalogSyncStatus,
    CatalogSummary,
    CatalogTraversalRequest,
    CatalogTraversalResult,
    FacetKind,
    RelationshipDirection,
    RelationshipFieldPair,
    RelationshipKind,
    RelationshipProvenance,
    ResourceKind,
    Sensitivity,
    SourceCatalogSnapshot,
)
from ..identity import AgentIdentity, AgentIdentityConflictError
from ..learning_candidates import (
    DocumentCandidateContent,
    LEARNING_CANDIDATE_MAX_RECORDS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_STAMPS,
    LearningCandidate,
    LearningCandidateAction,
    LearningCandidateError,
    LearningCandidateNotFoundError,
    LearningCandidateRejectionReason,
    LearningCandidateReviewStamp,
    LearningCandidateRunReference,
    LearningCandidateStatus,
    LearningCandidateTarget,
    LearningReviewRunTail,
    SemanticCandidateContent,
    SkillCandidateContent,
)
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from ..llm.pricing import (
    CostBasis,
    CostComponent,
    CostEstimate,
    CostEstimateStatus,
    PricingModifier,
    PricingUsageRange,
)
from ..loop.models import ConversationRun, LoopExit, LoopExitKind, RunInput, Transcript
from ..semantics import (
    SEMANTIC_MAX_ANNOTATIONS,
    ResourceRevisionBinding,
    SemanticAnnotation,
    SemanticDigestMismatchError,
    SemanticEvidence,
    SemanticEvidenceKind,
    SemanticFieldReference,
    SemanticKind,
    SemanticNotFoundError,
    SemanticSubject,
    SemanticValidationError,
    semantic_annotation_sha256,
)

_SEARCH_TERM = re.compile(r"[A-Za-z0-9_]+")
_ACTIVE_SOURCE_KEY_PREFIX = "active_source:"
_LEARNING_REVIEW_STAMPS_KEY_PREFIX = "learning_review_stamps:"
_T = TypeVar("_T")
_V1_TABLE_DEFINITIONS = {
    "learning_candidates": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("data", "TEXT", 1, None, 0),
    ),
    "messages": (
        ("run_id", "TEXT", 1, None, 1),
        ("position", "INTEGER", 1, None, 2),
        ("data", "TEXT", 1, None, 0),
    ),
    "metadata": (
        ("key", "TEXT", 0, None, 1),
        ("data", "TEXT", 1, None, 0),
    ),
    "runs": (
        ("id", "TEXT", 0, None, 1),
        ("agent_id", "TEXT", 1, None, 0),
        ("conversation_id", "TEXT", 1, None, 0),
        ("turn_index", "INTEGER", 1, None, 0),
        ("input", "TEXT", 1, None, 0),
        ("result", "TEXT", 0, None, 0),
    ),
    "semantic_annotations": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("data", "TEXT", 1, None, 0),
    ),
    "snapshots": (
        ("agent_id", "TEXT", 1, None, 1),
        ("source_id", "TEXT", 1, None, 2),
        ("sync_id", "TEXT", 1, None, 0),
        ("data", "TEXT", 1, None, 0),
    ),
    "sources": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("data", "TEXT", 1, None, 0),
    ),
    "syncs": (
        ("agent_id", "TEXT", 1, None, 1),
        ("id", "TEXT", 1, None, 2),
        ("source_id", "TEXT", 1, None, 0),
        ("data", "TEXT", 1, None, 0),
    ),
}
_V1_MESSAGES_FOREIGN_KEYS = (("runs", "run_id", "id", "NO ACTION", "CASCADE", "NONE"),)
_V1_NAMED_INDEXES = {
    "runs_conversation_turn": (
        "runs",
        True,
        ("agent_id", "conversation_id", "turn_index"),
    )
}
_INCOMPATIBLE_STATE_MESSAGE = (
    "state database is not compatible with this Daita release; "
    "the agent home was preserved"
)


def _active_source_key(agent_id: str) -> str:
    return f"{_ACTIVE_SOURCE_KEY_PREFIX}{agent_id}"


def _learning_review_stamps_key(agent_id: str) -> str:
    return f"{_LEARNING_REVIEW_STAMPS_KEY_PREFIX}{agent_id}"


class _CatalogCommitGate:
    """Linearize task cancellation against one SQLite mutation transaction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._started = False

    def start(self, connection: sqlite3.Connection) -> bool:
        with self._lock:
            if self._cancelled:
                return False
        connection.execute("BEGIN IMMEDIATE")
        with self._lock:
            if self._cancelled:
                connection.rollback()
                return False
            self._started = True
            return True

    def cancel_before_start(self) -> bool:
        with self._lock:
            if self._started:
                return False
            self._cancelled = True
            return True


class SQLiteStateStore:
    """One fixed-format persistence boundary for the launched v1 agent home."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @classmethod
    async def open(cls, path: str | Path, **_: object) -> SQLiteStateStore:
        resolved = Path(path).resolve()
        await asyncio.to_thread(_initialize, resolved)
        return cls(resolved)

    async def close(self) -> None:
        return None

    async def initialize_identity(self, identity: AgentIdentity) -> AgentIdentity:
        def write() -> AgentIdentity:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
                if row is not None:
                    current = _expect(_loads(row[0]), AgentIdentity)
                    if current != identity:
                        raise AgentIdentityConflictError(
                            "state database already belongs to another agent"
                        )
                    return current
                connection.execute(
                    "INSERT INTO metadata(key, data) VALUES ('identity', ?)",
                    (_dumps(identity),),
                )
                return identity

        return await asyncio.to_thread(write)

    async def load_identity(self) -> AgentIdentity | None:
        def read() -> AgentIdentity | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), AgentIdentity)

        return await asyncio.to_thread(read)

    async def register_source(
        self, registration: SourceRegistration
    ) -> SourceRegistration:
        def write() -> SourceRegistration:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (registration.agent_id, registration.id),
                ).fetchone()
                if row is not None:
                    current = _expect(_loads(row[0]), SourceRegistration)
                    if current != registration:
                        raise ValueError(
                            f"source registration already exists: {registration.id}"
                        )
                    return current
                connection.execute(
                    "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                    (registration.agent_id, registration.id, _dumps(registration)),
                )
                return registration

        return await asyncio.to_thread(write)

    async def load_source(
        self, agent_id: str, source_id: str
    ) -> SourceRegistration | None:
        def read() -> SourceRegistration | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (agent_id, source_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), SourceRegistration)

        return await asyncio.to_thread(read)

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        def read() -> tuple[SourceRegistration, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? ORDER BY id",
                    (agent_id,),
                ).fetchall()
            return tuple(_expect(_loads(row[0]), SourceRegistration) for row in rows)

        return await asyncio.to_thread(read)

    async def load_active_source_id(self, agent_id: str) -> str | None:
        def read() -> str | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = ?",
                    (_active_source_key(agent_id),),
                ).fetchone()
            if row is None:
                return None
            source_id = _loads(row[0])
            if not isinstance(source_id, str) or not source_id:
                raise ValueError("stored active source id is invalid")
            return source_id

        return await asyncio.to_thread(read)

    async def set_active_source_id(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceRegistration:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("source_id must be a non-empty string")

        def write() -> SourceRegistration:
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (agent_id, source_id),
                ).fetchone()
                if row is None:
                    raise ValueError("unknown active source for this agent")
                registration = _expect(_loads(row[0]), SourceRegistration)
                if not registration.active:
                    raise ValueError("unknown active source for this agent")
                connection.execute(
                    """INSERT INTO metadata(key, data) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                    (_active_source_key(agent_id), _dumps(source_id)),
                )
                return registration

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        registration = worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return registration

    async def detach_source(
        self, agent_id: str, source_id: str, detached_at: datetime
    ) -> SourceRegistration:
        gate = _CatalogCommitGate()

        def write() -> SourceRegistration | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                row = connection.execute(
                    "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                    (agent_id, source_id),
                ).fetchone()
                if row is None:
                    raise KeyError(f"unknown source: {source_id}")
                current = _expect(_loads(row[0]), SourceRegistration)
                detached = (
                    current if not current.active else current.detach(detached_at)
                )
                connection.execute(
                    "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
                    (_dumps(detached), agent_id, source_id),
                )
                selection = connection.execute(
                    "SELECT data FROM metadata WHERE key = ?",
                    (_active_source_key(agent_id),),
                ).fetchone()
                selected_id = None if selection is None else _loads(selection[0])
                if selected_id == source_id:
                    connection.execute(
                        "DELETE FROM metadata WHERE key = ?",
                        (_active_source_key(agent_id),),
                    )
                    remaining: list[SourceRegistration] = []
                    for (data,) in connection.execute(
                        "SELECT data FROM sources WHERE agent_id = ? ORDER BY id",
                        (agent_id,),
                    ).fetchall():
                        candidate = _expect(_loads(data), SourceRegistration)
                        if candidate.active:
                            remaining.append(candidate)
                    if len(remaining) == 1:
                        connection.execute(
                            "INSERT INTO metadata(key, data) VALUES (?, ?)",
                            (
                                _active_source_key(agent_id),
                                _dumps(remaining[0].id),
                            ),
                        )
                connection.commit()
                return detached
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        detached = worker.result()
        if cancelled_before_start:
            if detached is not None:
                raise AssertionError("cancelled source detach transaction committed")
            raise asyncio.CancelledError
        if detached is None:
            raise AssertionError(
                "source detach transaction stopped without cancellation"
            )
        return detached

    async def record_sync(self, sync: CatalogSync) -> CatalogSync:
        def write() -> CatalogSync:
            with _connect(self.path) as connection:
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (sync.agent_id, sync.id, sync.source_id, _dumps(sync)),
                )
                return sync

        return await asyncio.to_thread(write)

    async def commit_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
        *,
        registration: SourceRegistration | None = None,
    ) -> SourceCatalogSnapshot:
        sync = snapshot.sync
        if registration is not None and (
            registration.agent_id != sync.agent_id or registration.id != sync.source_id
        ):
            raise ValueError("catalog snapshot and source registration disagree")
        gate = _CatalogCommitGate()

        def write() -> SourceCatalogSnapshot | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                if registration is not None:
                    row = connection.execute(
                        "SELECT data FROM sources WHERE agent_id = ? AND id = ?",
                        (registration.agent_id, registration.id),
                    ).fetchone()
                    if row is None:
                        connection.execute(
                            "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                            (
                                registration.agent_id,
                                registration.id,
                                _dumps(registration),
                            ),
                        )
                    else:
                        current = _expect(_loads(row[0]), SourceRegistration)
                        if current != registration:
                            raise ValueError(
                                "source registration already exists: "
                                f"{registration.id}"
                            )
                    selection = connection.execute(
                        "SELECT 1 FROM metadata WHERE key = ?",
                        (_active_source_key(registration.agent_id),),
                    ).fetchone()
                    if selection is None:
                        connection.execute(
                            "INSERT INTO metadata(key, data) VALUES (?, ?)",
                            (
                                _active_source_key(registration.agent_id),
                                _dumps(registration.id),
                            ),
                        )
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (sync.agent_id, sync.id, sync.source_id, _dumps(sync)),
                )
                connection.execute(
                    """INSERT INTO snapshots(agent_id, source_id, sync_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, source_id) DO UPDATE SET
                         sync_id = excluded.sync_id, data = excluded.data""",
                    (sync.agent_id, sync.source_id, sync.id, _dumps(snapshot)),
                )
                _commit_catalog_transaction(connection)
                return snapshot
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        committed = worker.result()
        if cancelled_before_start:
            if committed is not None:
                raise AssertionError("cancelled catalog transaction committed")
            raise asyncio.CancelledError
        if committed is None:
            raise AssertionError("catalog transaction stopped without cancellation")
        return committed

    async def load_sync(self, agent_id: str, sync_id: str) -> CatalogSync | None:
        def read() -> CatalogSync | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM syncs WHERE agent_id = ? AND id = ?",
                    (agent_id, sync_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), CatalogSync)

        return await asyncio.to_thread(read)

    async def summarize_catalog(
        self,
        agent_id: str,
        active_source_ids: tuple[str, ...],
    ) -> CatalogSummary:
        """Count only current committed snapshots for the supplied active sources."""

        if not isinstance(active_source_ids, tuple) or any(
            not isinstance(source_id, str) or not source_id
            for source_id in active_source_ids
        ):
            raise TypeError("active_source_ids must be a tuple of non-empty strings")
        if len(active_source_ids) != len(set(active_source_ids)):
            raise ValueError("active_source_ids cannot contain duplicates")
        selected_source_ids = frozenset(active_source_ids)

        def read() -> CatalogSummary:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    "SELECT source_id, data FROM snapshots "
                    "WHERE agent_id = ? ORDER BY source_id",
                    (agent_id,),
                ).fetchall()
            snapshots = tuple(
                _expect(_loads(data), SourceCatalogSnapshot)
                for source_id, data in rows
                if source_id in selected_source_ids
            )
            completion_times = tuple(
                snapshot.sync.completed_at
                for snapshot in snapshots
                if snapshot.sync.completed_at is not None
            )
            resource_count = sum(len(snapshot.resources) for snapshot in snapshots)
            return CatalogSummary(
                active_source_count=len(active_source_ids),
                resource_count=resource_count,
                relationship_count=sum(
                    len(snapshot.relationships) for snapshot in snapshots
                ),
                latest_successful_sync_completed_at=(
                    max(completion_times) if completion_times else None
                ),
                is_empty=resource_count == 0,
            )

        return await asyncio.to_thread(read)

    async def load_resource(
        self, agent_id: str, resource_id: str
    ) -> CatalogResource | None:
        for snapshot in await self._snapshots(agent_id):
            for resource in snapshot.resources:
                if resource.id == resource_id:
                    return resource
        return None

    async def load_revision(
        self, agent_id: str, resource_id: str, revision: str
    ) -> CatalogResourceRevision | None:
        for snapshot in await self._snapshots(agent_id):
            for item in snapshot.revisions:
                if item.resource_id == resource_id and item.revision == revision:
                    return item
        return None

    async def list_resources(
        self, agent_id: str, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        resources = [
            resource
            for snapshot in await self._snapshots(agent_id)
            if source_id is None or snapshot.sync.source_id == source_id
            for resource in snapshot.resources
        ]
        return tuple(sorted(resources, key=lambda item: (item.name, item.id)))

    async def load_facets(
        self,
        agent_id: str,
        resource_id: str,
        revision: str | None = None,
    ) -> tuple[CatalogFacet, ...]:
        for snapshot in await self._snapshots(agent_id):
            resource = next(
                (item for item in snapshot.resources if item.id == resource_id), None
            )
            if resource is None or (
                revision is not None and resource.current_revision != revision
            ):
                continue
            return tuple(
                facet for facet in snapshot.facets if facet.resource_id == resource_id
            )
        return ()

    async def load_incident_relationships(
        self,
        agent_id: str,
        resource_id: str,
        *,
        relationship_kinds: tuple[RelationshipKind, ...] = (),
        limit: int = 50,
    ) -> tuple[CatalogRelationship, ...]:
        relationships = [
            item
            for item in await self._relationships(agent_id)
            if resource_id in (item.from_resource_id, item.to_resource_id)
            and (not relationship_kinds or item.kind in relationship_kinds)
        ]
        return tuple(sorted(relationships, key=lambda item: item.id)[:limit])

    async def load_relationships(
        self, agent_id: str, relationship_ids: tuple[str, ...]
    ) -> tuple[CatalogRelationship, ...]:
        by_id = {item.id: item for item in await self._relationships(agent_id)}
        return tuple(by_id[item_id] for item_id in relationship_ids if item_id in by_id)

    async def search(self, request: CatalogSearchRequest) -> CatalogSearchResult:
        query = request.query.casefold().strip()
        terms = tuple(
            dict.fromkeys(term.casefold() for term in _SEARCH_TERM.findall(query))
        )
        needles = tuple(dict.fromkeys((query, *terms))) if query else ()
        snapshots = await self._snapshots(request.agent_id)
        resources = tuple(
            sorted(
                (resource for snapshot in snapshots for resource in snapshot.resources),
                key=lambda item: (item.name.casefold(), item.native_identity, item.id),
            )
        )
        facets_by_resource: dict[str, tuple[CatalogFacet, ...]] = {}
        relationships: list[CatalogRelationship] = []
        for snapshot in snapshots:
            relationships.extend(snapshot.relationships)
            for resource in snapshot.resources:
                facets_by_resource[resource.id] = tuple(
                    facet
                    for facet in snapshot.facets
                    if facet.resource_id == resource.id
                )
        relationship_fields: dict[str, set[str]] = {}
        adjacency: dict[str, list[tuple[str, CatalogRelationship]]] = {}
        for relationship in sorted(relationships, key=lambda item: item.id):
            adjacency.setdefault(relationship.from_resource_id, []).append(
                (relationship.to_resource_id, relationship)
            )
            adjacency.setdefault(relationship.to_resource_id, []).append(
                (relationship.from_resource_id, relationship)
            )
            for pair in relationship.field_pairs:
                relationship_fields.setdefault(
                    relationship.from_resource_id, set()
                ).add(pair.source_field)
                relationship_fields.setdefault(relationship.to_resource_id, set()).add(
                    pair.target_field
                )

        eligible_by_id: dict[str, CatalogResource] = {}
        direct: list[CatalogSearchHit] = []
        for resource in resources:
            if request.source_ids and resource.source_id not in request.source_ids:
                continue
            if request.resource_kinds and resource.kind not in request.resource_kinds:
                continue
            eligible_by_id[resource.id] = resource
            resource_names = (
                ("name", resource.name.casefold()),
                ("native_identity", resource.native_identity.casefold()),
            )
            structural_fields: list[tuple[str, str]] = []
            for facet in facets_by_resource.get(resource.id, ()):
                if facet.kind is not FacetKind.TABULAR:
                    continue
                raw_columns = facet.payload.get("columns", ())
                if isinstance(raw_columns, tuple):
                    for column in raw_columns:
                        if not isinstance(column, FrozenJsonObject):
                            continue
                        name = column.get("name")
                        if isinstance(name, str):
                            structural_fields.append(
                                (f"column:{name}", name.casefold())
                            )
                raw_indexes = facet.payload.get("indexes", ())
                if isinstance(raw_indexes, tuple):
                    for index in raw_indexes:
                        if not isinstance(index, FrozenJsonObject):
                            continue
                        columns = index.get("columns")
                        if not isinstance(columns, tuple):
                            continue
                        for name in columns:
                            if isinstance(name, str):
                                structural_fields.append(
                                    (f"index_field:{name}", name.casefold())
                                )
            structural_fields.extend(
                (f"relationship_field:{name}", name.casefold())
                for name in sorted(relationship_fields.get(resource.id, ()))
            )
            metadata_fields = (
                (
                    "kind",
                    f"{resource.kind.value} {resource.kind.value}s".casefold(),
                ),
                ("external_uri", resource.external_uri.casefold()),
            )
            reason, rank = _catalog_search_reason(
                needles,
                resource_names,
                tuple(structural_fields),
                metadata_fields,
            )
            if reason is None:
                continue
            matched_fields = tuple(
                dict.fromkeys(
                    name
                    for name, value in (
                        *resource_names,
                        *structural_fields,
                        *metadata_fields,
                    )
                    if any(needle in value for needle in needles)
                )
            )
            direct.append(
                CatalogSearchHit(
                    resource_id=resource.id,
                    source_id=resource.source_id,
                    kind=resource.kind,
                    name=resource.name,
                    revision=resource.current_revision,
                    sensitivity=resource.sensitivity,
                    score=float(600 - rank),
                    matched_fields=matched_fields[:32],
                    match_reasons=(reason,),
                )
            )
        direct.sort(
            key=lambda item: (
                -item.score,
                item.name.casefold(),
                item.name,
                item.resource_id,
            )
        )
        direct_ids = {item.resource_id for item in direct}
        neighbor_by_id: dict[str, CatalogSearchHit] = {}
        for hit in direct:
            for neighbor_id, relationship in adjacency.get(hit.resource_id, ()):
                neighbor = eligible_by_id.get(neighbor_id)
                if (
                    neighbor is None
                    or neighbor_id in direct_ids
                    or neighbor_id in neighbor_by_id
                ):
                    continue
                neighbor_by_id[neighbor_id] = CatalogSearchHit(
                    resource_id=neighbor.id,
                    source_id=neighbor.source_id,
                    kind=neighbor.kind,
                    name=neighbor.name,
                    revision=neighbor.current_revision,
                    sensitivity=neighbor.sensitivity,
                    score=0.0,
                    matched_fields=(f"relationship:{relationship.id}",),
                    match_reasons=("relationship_neighbor",),
                )
        neighbors = sorted(
            neighbor_by_id.values(),
            key=lambda item: (
                item.name.casefold(),
                item.name,
                item.resource_id,
            ),
        )
        matches = [*direct, *neighbors]
        hits = tuple(matches[: request.limit])
        return CatalogSearchResult(
            request=request,
            hits=hits,
            total_matches=len(matches),
            truncated=len(matches) > len(hits),
        )

    async def traverse(
        self, request: CatalogTraversalRequest
    ) -> CatalogTraversalResult:
        relationships = tuple(
            item
            for item in await self._relationships(request.agent_id)
            if not request.relationship_kinds or item.kind in request.relationship_kinds
        )
        adjacency: dict[
            str, list[tuple[str, CatalogRelationship, RelationshipDirection]]
        ] = {}
        for item in relationships:
            adjacency.setdefault(item.from_resource_id, []).append(
                (item.to_resource_id, item, RelationshipDirection.FORWARD)
            )
            adjacency.setdefault(item.to_resource_id, []).append(
                (item.from_resource_id, item, RelationshipDirection.REVERSE)
            )
        targets = set(request.to_resource_ids)
        queue: deque[tuple[str, tuple[str, ...], tuple[CatalogPathStep, ...]]] = deque(
            (source, (source,), ()) for source in request.from_resource_ids
        )
        paths: list[CatalogPath] = []
        visited_nodes: set[str] = set(request.from_resource_ids)
        visited_edges: set[str] = set()
        truncated = False
        while queue and len(paths) < request.max_paths:
            current, resource_ids, steps = queue.popleft()
            if current in targets and steps:
                paths.append(CatalogPath(resource_ids=resource_ids, steps=steps))
                continue
            if len(steps) >= request.max_depth:
                continue
            for neighbor, relationship, direction in adjacency.get(current, ()):
                if neighbor in resource_ids:
                    continue
                if len(visited_edges) >= request.max_edges or (
                    neighbor not in visited_nodes
                    and len(visited_nodes) >= request.max_nodes
                ):
                    truncated = True
                    continue
                visited_edges.add(relationship.id)
                visited_nodes.add(neighbor)
                queue.append(
                    (
                        neighbor,
                        (*resource_ids, neighbor),
                        (
                            *steps,
                            CatalogPathStep(
                                relationship_id=relationship.id,
                                from_resource_id=current,
                                to_resource_id=neighbor,
                                direction=direction,
                            ),
                        ),
                    )
                )
        if queue:
            truncated = True
        return CatalogTraversalResult(
            request=request,
            paths=tuple(paths),
            reachable=bool(paths),
            visited_nodes=len(visited_nodes),
            visited_edges=len(visited_edges),
            truncated=truncated,
        )

    async def list_semantic_annotations(
        self,
        agent_id: str,
    ) -> tuple[SemanticAnnotation, ...]:
        """Return one bounded deterministic agent-isolated semantic collection."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")

        def read() -> tuple[SemanticAnnotation, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT data FROM semantic_annotations
                       WHERE agent_id = ? ORDER BY id""",
                    (agent_id,),
                ).fetchall()
            if len(rows) > SEMANTIC_MAX_ANNOTATIONS:
                raise RuntimeError(
                    "stored semantic annotation collection exceeds bound"
                )
            return tuple(_expect(_loads(data), SemanticAnnotation) for (data,) in rows)

        return await asyncio.to_thread(read)

    async def load_semantic_annotation(
        self,
        agent_id: str,
        annotation_id: str,
    ) -> SemanticAnnotation | None:
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("annotation_id must be a non-empty string")

        def read() -> SemanticAnnotation | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM semantic_annotations
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, annotation_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), SemanticAnnotation)

        return await asyncio.to_thread(read)

    async def preflight_semantic_save(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        expected_sha256: str | None,
    ) -> FrozenJsonObject:
        """Validate and fingerprint one exact semantic create or replacement."""

        _validate_semantic_owner(agent_id, annotation)

        def read() -> FrozenJsonObject:
            with _connect(self.path) as connection:
                return _semantic_save_fingerprint(
                    connection,
                    agent_id,
                    annotation,
                    expected_sha256,
                )

        return await asyncio.to_thread(read)

    async def save_semantic_annotation(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        *,
        expected_sha256: str | None = None,
    ) -> bool:
        """Atomically create, digest-replace, or digest-supersede one annotation."""

        _validate_semantic_owner(agent_id, annotation)
        gate = _CatalogCommitGate()

        def write() -> bool | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                _semantic_save_fingerprint(
                    connection,
                    agent_id,
                    annotation,
                    expected_sha256,
                )
                row = connection.execute(
                    """SELECT data FROM semantic_annotations
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, annotation.id),
                ).fetchone()
                changed = (
                    row is None
                    or _expect(_loads(row[0]), SemanticAnnotation) != annotation
                )
                connection.execute(
                    """INSERT INTO semantic_annotations(agent_id, id, data)
                       VALUES (?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (agent_id, annotation.id, _dumps(annotation)),
                )
                connection.commit()
                return changed
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        changed = worker.result()
        if cancelled_before_start:
            if changed is not None:
                raise AssertionError("cancelled semantic transaction committed")
            raise asyncio.CancelledError
        if changed is None:
            raise AssertionError("semantic transaction stopped without cancellation")
        return changed

    async def preflight_semantic_delete(
        self,
        agent_id: str,
        annotation_id: str,
        expected_sha256: str,
    ) -> FrozenJsonObject:
        """Validate and fingerprint one digest-protected semantic deletion."""

        def read() -> FrozenJsonObject:
            with _connect(self.path) as connection:
                return _semantic_delete_fingerprint(
                    connection,
                    agent_id,
                    annotation_id,
                    expected_sha256,
                )

        return await asyncio.to_thread(read)

    async def delete_semantic_annotation(
        self,
        agent_id: str,
        annotation_id: str,
        *,
        expected_sha256: str,
    ) -> bool:
        """Atomically delete one annotation only when its rendered digest matches."""

        gate = _CatalogCommitGate()

        def write() -> bool | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                _semantic_delete_fingerprint(
                    connection,
                    agent_id,
                    annotation_id,
                    expected_sha256,
                )
                cursor = connection.execute(
                    """DELETE FROM semantic_annotations
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, annotation_id),
                )
                if cursor.rowcount != 1:
                    raise SemanticNotFoundError(annotation_id)
                connection.commit()
                return True
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        deleted = worker.result()
        if cancelled_before_start:
            if deleted is not None:
                raise AssertionError("cancelled semantic transaction committed")
            raise asyncio.CancelledError
        if deleted is None:
            raise AssertionError("semantic transaction stopped without cancellation")
        return deleted

    async def recent_completed_runs(
        self,
        agent_id: str,
        *,
        limit: int,
    ) -> LearningReviewRunTail:
        """Return a bounded chronological tail of terminal completed runs."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit < 1
            or limit > LEARNING_REVIEW_MAX_STAMPS
        ):
            raise ValueError("completed-run review limit is invalid")

        def read() -> LearningReviewRunTail:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT id, turn_index, input, result
                       FROM runs
                       WHERE agent_id = ? AND result IS NOT NULL
                       ORDER BY rowid DESC
                       LIMIT ?""",
                    (agent_id, limit),
                ).fetchall()
                records: list[ConversationRun] = []
                unreadable_run_count = 0
                for run_id, turn_index, input_data, result_data in rows:
                    try:
                        result = _expect(_loads(result_data), LoopExit)
                        if result.kind is not LoopExitKind.COMPLETED:
                            continue
                        message_rows = connection.execute(
                            """SELECT data FROM messages
                               WHERE run_id = ? ORDER BY position""",
                            (run_id,),
                        ).fetchall()
                        record = ConversationRun(
                            turn_index=int(turn_index),
                            transcript=Transcript(
                                run=_expect(_loads(input_data), RunInput),
                                messages=tuple(
                                    _expect(_loads(data), CanonicalMessage)
                                    for (data,) in message_rows
                                ),
                            ),
                            result=result,
                        )
                    except (KeyError, TypeError, ValueError):
                        unreadable_run_count += 1
                        continue
                    records.append(record)
            records.reverse()
            return LearningReviewRunTail(
                tuple(records),
                unreadable_run_count=unreadable_run_count,
            )

        return await asyncio.to_thread(read)

    async def list_learning_candidates(
        self,
        agent_id: str,
    ) -> tuple[LearningCandidate, ...]:
        """Return one bounded deterministic agent-isolated candidate inbox."""

        _validate_learning_agent_id(agent_id)

        def read() -> tuple[LearningCandidate, ...]:
            with _connect(self.path) as connection:
                return _learning_candidate_rows(connection, agent_id)

        return await asyncio.to_thread(read)

    async def load_learning_candidate(
        self,
        agent_id: str,
        candidate_id: str,
    ) -> LearningCandidate | None:
        _validate_learning_agent_id(agent_id)
        _validate_learning_candidate_id(candidate_id)

        def read() -> LearningCandidate | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM learning_candidates
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, candidate_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), LearningCandidate)

        return await asyncio.to_thread(read)

    async def learning_candidate_review_stamps(
        self,
        agent_id: str,
    ) -> tuple[LearningCandidateReviewStamp, ...]:
        _validate_learning_agent_id(agent_id)

        def read() -> tuple[LearningCandidateReviewStamp, ...]:
            with _connect(self.path) as connection:
                return _learning_review_stamps(connection, agent_id)

        return await asyncio.to_thread(read)

    async def save_learning_candidate_review(
        self,
        agent_id: str,
        *,
        stamps: tuple[LearningCandidateReviewStamp, ...],
        candidates: tuple[LearningCandidate, ...],
    ) -> tuple[LearningCandidate, ...]:
        """Atomically record one completed review and its inactive candidates."""

        _validate_learning_agent_id(agent_id)
        stamps = tuple(stamps)
        candidates = tuple(candidates)
        if (
            not stamps
            or len(stamps) > LEARNING_REVIEW_MAX_STAMPS
            or any(
                not isinstance(item, LearningCandidateReviewStamp) for item in stamps
            )
            or len(set(stamps)) != len(stamps)
        ):
            raise LearningCandidateError("review stamps exceed their unique bound")
        if len(candidates) > LEARNING_REVIEW_MAX_PROPOSALS or any(
            not isinstance(item, LearningCandidate) for item in candidates
        ):
            raise LearningCandidateError("review candidates exceed their bound")
        for candidate in candidates:
            _validate_learning_candidate_owner(agent_id, candidate)
            if candidate.status is not LearningCandidateStatus.AWAITING_REVIEW:
                raise LearningCandidateError(
                    "new review candidates must be awaiting review"
                )

        def write(connection: sqlite3.Connection) -> tuple[LearningCandidate, ...]:
            current_stamps = _learning_review_stamps(connection, agent_id)
            current_stamp_set = set(current_stamps)
            if all(stamp in current_stamp_set for stamp in stamps):
                review_fingerprints = {item.review_fingerprint for item in candidates}
                return tuple(
                    item
                    for item in _learning_candidate_rows(connection, agent_id)
                    if item.review_fingerprint in review_fingerprints
                )
            merged_stamps = (
                *current_stamps,
                *(stamp for stamp in stamps if stamp not in current_stamp_set),
            )
            if len(merged_stamps) > LEARNING_REVIEW_MAX_STAMPS:
                raise LearningCandidateError(
                    "learning review stamp capacity is exhausted"
                )
            current = _learning_candidate_rows(connection, agent_id)
            by_id = {item.id: item for item in current}
            identities = {item.candidate_identity_sha256 for item in current}
            inserted: list[LearningCandidate] = []
            for candidate in candidates:
                existing = by_id.get(candidate.id)
                if existing is not None:
                    if (
                        existing.candidate_identity_sha256
                        == candidate.candidate_identity_sha256
                    ):
                        continue
                    raise LearningCandidateError(
                        "learning candidate record identity collision"
                    )
                if candidate.candidate_identity_sha256 in identities:
                    continue
                if len(current) + len(inserted) >= LEARNING_CANDIDATE_MAX_RECORDS:
                    raise LearningCandidateError(
                        "learning candidate capacity is exhausted"
                    )
                connection.execute(
                    """INSERT INTO learning_candidates(agent_id, id, data)
                       VALUES (?, ?, ?)""",
                    (agent_id, candidate.id, _dumps(candidate)),
                )
                by_id[candidate.id] = candidate
                identities.add(candidate.candidate_identity_sha256)
                inserted.append(candidate)
            connection.execute(
                """INSERT INTO metadata(key, data) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                (
                    _learning_review_stamps_key(agent_id),
                    _dumps(tuple(merged_stamps)),
                ),
            )
            return tuple(inserted)

        return await _run_candidate_transaction(self.path, write)

    async def edit_learning_candidate(
        self,
        agent_id: str,
        candidate: LearningCandidate,
        *,
        expected_fingerprint: str,
    ) -> LearningCandidate:
        """Atomically replace only one awaiting candidate's bounded content."""

        _validate_learning_agent_id(agent_id)
        _validate_learning_candidate_owner(agent_id, candidate)
        _validate_learning_digest(expected_fingerprint, "expected_fingerprint")

        def write(connection: sqlite3.Connection) -> LearningCandidate:
            current = _load_learning_candidate_required(
                connection,
                agent_id,
                candidate.id,
            )
            _require_candidate_transition(
                current,
                expected_fingerprint=expected_fingerprint,
            )
            if candidate.status is not LearningCandidateStatus.AWAITING_REVIEW:
                raise LearningCandidateError(
                    "edited candidate must remain awaiting review"
                )
            unchanged = (
                "id",
                "agent_id",
                "target",
                "source_ids",
                "reviewed_runs",
                "supporting_run_ids",
                "review_fingerprint",
                "artifact_state_sha256",
                "catalog_revisions",
                "status",
                "created_at",
                "rejection_reason",
            )
            if any(
                getattr(current, field_name) != getattr(candidate, field_name)
                for field_name in unchanged
            ):
                raise LearningCandidateError(
                    "candidate edit may change only bounded proposed content"
                )
            if candidate.updated_at < current.updated_at:
                raise LearningCandidateError(
                    "candidate edit timestamp cannot move backwards"
                )
            if any(
                item.id != candidate.id
                and item.candidate_identity_sha256
                == candidate.candidate_identity_sha256
                for item in _learning_candidate_rows(connection, agent_id)
            ):
                raise LearningCandidateError(
                    "candidate edit duplicates an existing normalized identity"
                )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (_dumps(candidate), agent_id, candidate.id),
            )
            return candidate

        return await _run_candidate_transaction(self.path, write)

    async def reject_learning_candidate(
        self,
        agent_id: str,
        candidate_id: str,
        *,
        expected_fingerprint: str,
        reason: LearningCandidateRejectionReason,
        rejected_at: datetime,
    ) -> LearningCandidate:
        _validate_learning_agent_id(agent_id)
        _validate_learning_candidate_id(candidate_id)
        _validate_learning_digest(expected_fingerprint, "expected_fingerprint")
        if not isinstance(reason, LearningCandidateRejectionReason):
            raise TypeError("candidate rejection reason is invalid")

        def write(connection: sqlite3.Connection) -> LearningCandidate:
            current = _load_learning_candidate_required(
                connection,
                agent_id,
                candidate_id,
            )
            _require_candidate_transition(
                current,
                expected_fingerprint=expected_fingerprint,
            )
            rejected = LearningCandidate(
                **{
                    **{
                        field.name: getattr(current, field.name)
                        for field in fields(current)
                    },
                    "status": LearningCandidateStatus.REJECTED,
                    "updated_at": rejected_at,
                    "rejection_reason": reason,
                }
            )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (_dumps(rejected), agent_id, candidate_id),
            )
            return rejected

        return await _run_candidate_transaction(self.path, write)

    async def accept_learning_candidate(
        self,
        agent_id: str,
        candidate_id: str,
        *,
        expected_fingerprint: str,
        accepted_at: datetime,
    ) -> LearningCandidate:
        _validate_learning_agent_id(agent_id)
        _validate_learning_candidate_id(candidate_id)
        _validate_learning_digest(expected_fingerprint, "expected_fingerprint")

        def write(connection: sqlite3.Connection) -> LearningCandidate:
            current = _load_learning_candidate_required(
                connection,
                agent_id,
                candidate_id,
            )
            _require_candidate_transition(
                current,
                expected_fingerprint=expected_fingerprint,
            )
            accepted = LearningCandidate(
                **{
                    **{
                        field.name: getattr(current, field.name)
                        for field in fields(current)
                    },
                    "status": LearningCandidateStatus.ACCEPTED,
                    "updated_at": accepted_at,
                }
            )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (_dumps(accepted), agent_id, candidate_id),
            )
            return accepted

        return await _run_candidate_transaction(self.path, write)

    async def clear_rejected_learning_candidates(self, agent_id: str) -> int:
        """Delete only explicit rejection tombstones and reset their review stamps."""

        _validate_learning_agent_id(agent_id)

        def write(connection: sqlite3.Connection) -> int:
            rejected_ids = tuple(
                item.id
                for item in _learning_candidate_rows(connection, agent_id)
                if item.status is LearningCandidateStatus.REJECTED
            )
            if rejected_ids:
                connection.executemany(
                    """DELETE FROM learning_candidates
                       WHERE agent_id = ? AND id = ?""",
                    ((agent_id, candidate_id) for candidate_id in rejected_ids),
                )
                connection.execute(
                    "DELETE FROM metadata WHERE key = ?",
                    (_learning_review_stamps_key(agent_id),),
                )
            return len(rejected_ids)

        return await _run_candidate_transaction(self.path, write)

    async def start(self, run: RunInput) -> Transcript:
        if run.conversation_id is None:
            raise ValueError("run conversation_id must be resolved before persistence")

        def write() -> Transcript:
            with _connect(self.path) as connection:
                try:
                    connection.execute("BEGIN IMMEDIATE")
                    row = connection.execute(
                        """SELECT COALESCE(MAX(turn_index), -1) + 1
                           FROM runs
                           WHERE agent_id = ? AND conversation_id = ?""",
                        (run.agent_id, run.conversation_id),
                    ).fetchone()
                    connection.execute(
                        """INSERT INTO runs(
                               id, agent_id, conversation_id, turn_index, input
                           ) VALUES (?, ?, ?, ?, ?)""",
                        (
                            run.id,
                            run.agent_id,
                            run.conversation_id,
                            int(row[0]),
                            _dumps(run),
                        ),
                    )
                except sqlite3.IntegrityError as error:
                    raise ValueError(f"run already exists: {run.id}") from error
            return Transcript(run=run)

        return await asyncio.to_thread(write)

    async def append(self, run_id: str, message: CanonicalMessage) -> None:
        def write() -> None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM messages WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if (
                    connection.execute(
                        "SELECT 1 FROM runs WHERE id = ?", (run_id,)
                    ).fetchone()
                    is None
                ):
                    raise KeyError(f"unknown run: {run_id}")
                connection.execute(
                    "INSERT INTO messages(run_id, position, data) VALUES (?, ?, ?)",
                    (run_id, int(row[0]), _dumps(message)),
                )

        await asyncio.to_thread(write)

    async def finish(self, result: LoopExit) -> None:
        def write() -> None:
            with _connect(self.path) as connection:
                cursor = connection.execute(
                    """UPDATE runs SET result = ?
                       WHERE id = ? AND conversation_id = ?""",
                    (_dumps(result), result.run_id, result.conversation_id),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown run: {result.run_id}")

        await asyncio.to_thread(write)

    async def load(self, run_id: str) -> Transcript:
        def read() -> Transcript:
            with _connect(self.path) as connection:
                run_row = connection.execute(
                    "SELECT input FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
                if run_row is None:
                    raise KeyError(f"unknown run: {run_id}")
                rows = connection.execute(
                    "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
                    (run_id,),
                ).fetchall()
            return Transcript(
                run=_expect(_loads(run_row[0]), RunInput),
                messages=tuple(
                    _expect(_loads(row[0]), CanonicalMessage) for row in rows
                ),
            )

        return await asyncio.to_thread(read)

    async def result(self, run_id: str) -> LoopExit | None:
        def read() -> LoopExit | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT result FROM runs WHERE id = ?", (run_id,)
                ).fetchone()
            if row is None:
                raise KeyError(f"unknown run: {run_id}")
            return None if row[0] is None else _expect(_loads(row[0]), LoopExit)

        return await asyncio.to_thread(read)

    async def conversation_runs(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        """Return every run in one agent-scoped conversation in turn order."""

        def read() -> tuple[ConversationRun, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT id, turn_index, input, result
                       FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       ORDER BY turn_index""",
                    (agent_id, conversation_id),
                ).fetchall()
                records: list[ConversationRun] = []
                for run_id, turn_index, input_data, result_data in rows:
                    message_rows = connection.execute(
                        """SELECT data FROM messages
                           WHERE run_id = ? ORDER BY position""",
                        (run_id,),
                    ).fetchall()
                    transcript = Transcript(
                        run=_expect(_loads(input_data), RunInput),
                        messages=tuple(
                            _expect(_loads(message[0]), CanonicalMessage)
                            for message in message_rows
                        ),
                    )
                    result = (
                        None
                        if result_data is None
                        else _expect(_loads(result_data), LoopExit)
                    )
                    records.append(
                        ConversationRun(
                            turn_index=int(turn_index),
                            transcript=transcript,
                            result=result,
                        )
                    )
            return tuple(records)

        return await asyncio.to_thread(read)

    async def conversation_exists(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> bool:
        """Return one agent-scoped existence fact without loading transcript data."""

        def read() -> bool:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT 1 FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       LIMIT 1""",
                    (agent_id, conversation_id),
                ).fetchone()
            return row is not None

        return await asyncio.to_thread(read)

    async def clear_conversations(self, agent_id: str) -> int:
        """Delete transcripts and candidate records derived from them."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        gate = _CatalogCommitGate()

        def write() -> int | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                row = connection.execute(
                    "SELECT COUNT(*) FROM runs WHERE agent_id = ?",
                    (agent_id,),
                ).fetchone()
                run_count = int(row[0])
                connection.execute(
                    """DELETE FROM messages
                       WHERE run_id IN (
                           SELECT id FROM runs WHERE agent_id = ?
                       )""",
                    (agent_id,),
                )
                connection.execute(
                    "DELETE FROM runs WHERE agent_id = ?",
                    (agent_id,),
                )
                connection.execute(
                    "DELETE FROM learning_candidates WHERE agent_id = ?",
                    (agent_id,),
                )
                connection.execute(
                    "DELETE FROM metadata WHERE key = ?",
                    (_learning_review_stamps_key(agent_id),),
                )
                connection.commit()
                return run_count
            except BaseException:
                connection.rollback()
                raise
            finally:
                connection.close()

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled_before_start = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled_before_start = (
                    gate.cancel_before_start() or cancelled_before_start
                )
        cleared = worker.result()
        if cancelled_before_start:
            if cleared is not None:
                raise AssertionError(
                    "cancelled conversation-clear transaction committed"
                )
            raise asyncio.CancelledError
        if cleared is None:
            raise AssertionError(
                "conversation-clear transaction stopped without cancellation"
            )
        return cleared

    async def conversation_source_id(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> str | None:
        """Return the sticky source captured by the first run in a conversation."""

        def read() -> str | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT input FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       ORDER BY turn_index
                       LIMIT 1""",
                    (agent_id, conversation_id),
                ).fetchone()
            if row is None:
                return None
            run = _expect(_loads(row[0]), RunInput)
            return run.conversation_source_id or run.source_id

        return await asyncio.to_thread(read)

    async def completed_conversation_tail(
        self,
        agent_id: str,
        conversation_id: str,
        *,
        limit: int = 8,
    ) -> tuple[bool, tuple[ConversationRun, ...], bool]:
        """Load one bounded newest-first candidate snapshot and existence fact."""

        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("conversation tail limit must be positive")

        def read() -> tuple[bool, tuple[ConversationRun, ...], bool]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT id, turn_index, input, result
                       FROM runs
                       WHERE agent_id = ? AND conversation_id = ?
                       ORDER BY turn_index DESC""",
                    (agent_id, conversation_id),
                )
                exists = False
                older_completed_exists = False
                records: list[ConversationRun] = []
                for run_id, turn_index, input_data, result_data in rows:
                    exists = True
                    if result_data is None:
                        continue
                    result = _expect(_loads(result_data), LoopExit)
                    if result.kind is not LoopExitKind.COMPLETED:
                        continue
                    if len(records) >= limit:
                        older_completed_exists = True
                        break
                    message_rows = connection.execute(
                        """SELECT data FROM messages
                           WHERE run_id = ? ORDER BY position""",
                        (run_id,),
                    ).fetchall()
                    records.append(
                        ConversationRun(
                            turn_index=int(turn_index),
                            transcript=Transcript(
                                run=_expect(_loads(input_data), RunInput),
                                messages=tuple(
                                    _expect(_loads(message[0]), CanonicalMessage)
                                    for message in message_rows
                                ),
                            ),
                            result=result,
                        )
                    )
            records.reverse()
            return exists, tuple(records), older_completed_exists

        return await asyncio.to_thread(read)

    async def _snapshots(self, agent_id: str) -> tuple[SourceCatalogSnapshot, ...]:
        def read() -> tuple[SourceCatalogSnapshot, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    "SELECT data FROM snapshots WHERE agent_id = ? ORDER BY source_id",
                    (agent_id,),
                ).fetchall()
            return tuple(_expect(_loads(row[0]), SourceCatalogSnapshot) for row in rows)

        return await asyncio.to_thread(read)

    async def _relationships(self, agent_id: str) -> tuple[CatalogRelationship, ...]:
        return tuple(
            relationship
            for snapshot in await self._snapshots(agent_id)
            for relationship in snapshot.relationships
        )


def _catalog_search_reason(
    needles: tuple[str, ...],
    resource_names: tuple[tuple[str, str], ...],
    structural_fields: tuple[tuple[str, str], ...],
    metadata_fields: tuple[tuple[str, str], ...],
) -> tuple[str | None, int]:
    """Return the strongest deterministic lexical match category."""

    if not needles:
        return "metadata_contains", 5
    for reason, rank, fields, predicate in (
        (
            "resource_name_exact",
            0,
            resource_names,
            lambda needle, value: needle == value,
        ),
        (
            "resource_name_prefix",
            1,
            resource_names,
            lambda needle, value: value.startswith(needle),
        ),
        (
            "resource_name_contains",
            2,
            resource_names,
            lambda needle, value: needle in value,
        ),
        (
            "structural_field_exact",
            3,
            structural_fields,
            lambda needle, value: needle == value,
        ),
        (
            "structural_field_contains",
            4,
            structural_fields,
            lambda needle, value: needle in value,
        ),
        (
            "metadata_contains",
            5,
            metadata_fields,
            lambda needle, value: needle in value,
        ),
    ):
        if any(predicate(needle, value) for needle in needles for _, value in fields):
            return reason, rank
    return None, 6


def _validate_semantic_owner(
    agent_id: str,
    annotation: SemanticAnnotation,
) -> None:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    if not isinstance(annotation, SemanticAnnotation):
        raise TypeError("annotation must be SemanticAnnotation")
    if annotation.agent_id != agent_id:
        raise ValueError("semantic annotation belongs to another agent")


def _semantic_rows(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[tuple[str, SemanticAnnotation, str], ...]:
    rows = connection.execute(
        """SELECT id, data FROM semantic_annotations
           WHERE agent_id = ? ORDER BY id""",
        (agent_id,),
    ).fetchall()
    if len(rows) > SEMANTIC_MAX_ANNOTATIONS:
        raise RuntimeError("stored semantic annotation collection exceeds bound")
    return tuple(
        (annotation_id, _expect(_loads(data), SemanticAnnotation), data)
        for annotation_id, data in rows
    )


def _semantic_state_sha256(
    rows: tuple[tuple[str, SemanticAnnotation, str], ...],
) -> str:
    payload = "\n".join(f"{annotation_id}:{data}" for annotation_id, _, data in rows)
    return sha256(payload.encode("utf-8")).hexdigest()


def _semantic_save_fingerprint(
    connection: sqlite3.Connection,
    agent_id: str,
    annotation: SemanticAnnotation,
    expected_sha256: str | None,
) -> FrozenJsonObject:
    _validate_semantic_owner(agent_id, annotation)
    if expected_sha256 is not None and (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise SemanticDigestMismatchError(
            "semantic expected_sha256 must be lowercase SHA-256"
        )
    rows = _semantic_rows(connection, agent_id)
    by_id = {annotation_id: item for annotation_id, item, _ in rows}
    current = by_id.get(annotation.id)
    expected_target: SemanticAnnotation | None
    expected_target_id: str
    if current is not None:
        expected_target = current
        expected_target_id = current.id
        if annotation.created_at != current.created_at:
            raise SemanticValidationError(
                "semantic replacement must preserve created_at"
            )
    elif annotation.supersedes_id is not None:
        expected_target = by_id.get(annotation.supersedes_id)
        expected_target_id = annotation.supersedes_id
        if expected_target is None:
            raise SemanticNotFoundError(annotation.supersedes_id)
        if (
            expected_target.subject != annotation.subject
            or expected_target.kind is not annotation.kind
        ):
            raise SemanticValidationError(
                "a semantic annotation may supersede only the same subject and kind"
            )
        if annotation.created_at < expected_target.created_at:
            raise SemanticValidationError(
                "semantic supersession cannot predate its target"
            )
    else:
        expected_target = None
        expected_target_id = annotation.id

    if expected_target is None:
        if expected_sha256 is not None:
            raise SemanticDigestMismatchError(
                "new semantic annotations cannot include expected_sha256"
            )
        if len(rows) >= SEMANTIC_MAX_ANNOTATIONS:
            raise SemanticValidationError(
                f"semantic annotation collection is limited to "
                f"{SEMANTIC_MAX_ANNOTATIONS}"
            )
        current_sha256 = sha256(b"").hexdigest()
    else:
        current_sha256 = semantic_annotation_sha256(expected_target)
        if expected_sha256 is None:
            raise SemanticDigestMismatchError(
                "semantic replacement or supersession requires expected_sha256"
            )
        if expected_sha256 != current_sha256:
            raise SemanticDigestMismatchError(
                "semantic annotation changed; load it again with semantic_view"
            )

    return FrozenJsonObject.from_mapping(
        {
            "id": annotation.id,
            "exists": current is not None,
            "expected_target_id": expected_target_id,
            "current_sha256": current_sha256,
            "candidate_sha256": semantic_annotation_sha256(annotation),
            "state_sha256": _semantic_state_sha256(rows),
        }
    )


def _semantic_delete_fingerprint(
    connection: sqlite3.Connection,
    agent_id: str,
    annotation_id: str,
    expected_sha256: str,
) -> FrozenJsonObject:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("agent_id must be a non-empty string")
    if not isinstance(annotation_id, str) or not annotation_id:
        raise ValueError("annotation_id must be a non-empty string")
    if (
        not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise SemanticDigestMismatchError(
            "semantic deletion requires lowercase expected_sha256"
        )
    rows = _semantic_rows(connection, agent_id)
    current = next(
        (item for item_id, item, _ in rows if item_id == annotation_id),
        None,
    )
    if current is None:
        raise SemanticNotFoundError(annotation_id)
    current_sha256 = semantic_annotation_sha256(current)
    if current_sha256 != expected_sha256:
        raise SemanticDigestMismatchError(
            "semantic annotation changed; load it again with semantic_view"
        )
    return FrozenJsonObject.from_mapping(
        {
            "id": annotation_id,
            "current_sha256": current_sha256,
            "state_sha256": _semantic_state_sha256(rows),
        }
    )


def _validate_learning_agent_id(agent_id: str) -> None:
    if not isinstance(agent_id, str) or not agent_id:
        raise ValueError("learning candidate agent_id must be non-empty text")


def _validate_learning_candidate_id(candidate_id: str) -> None:
    if not isinstance(candidate_id, str) or not candidate_id:
        raise ValueError("learning candidate id must be non-empty text")


def _validate_learning_digest(value: str, field_name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise LearningCandidateError(f"{field_name} must be lowercase SHA-256")


def _validate_learning_candidate_owner(
    agent_id: str,
    candidate: LearningCandidate,
) -> None:
    if not isinstance(candidate, LearningCandidate):
        raise TypeError("candidate must be LearningCandidate")
    if candidate.agent_id != agent_id:
        raise LearningCandidateError("learning candidate belongs to another agent")


def _learning_candidate_rows(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[LearningCandidate, ...]:
    rows = connection.execute(
        """SELECT data FROM learning_candidates
           WHERE agent_id = ? ORDER BY id""",
        (agent_id,),
    ).fetchall()
    if len(rows) > LEARNING_CANDIDATE_MAX_RECORDS:
        raise RuntimeError("stored learning candidate collection exceeds bound")
    values = tuple(_expect(_loads(data), LearningCandidate) for (data,) in rows)
    if any(item.agent_id != agent_id for item in values):
        raise RuntimeError("stored learning candidate owner is invalid")
    return values


def _learning_review_stamps(
    connection: sqlite3.Connection,
    agent_id: str,
) -> tuple[LearningCandidateReviewStamp, ...]:
    row = connection.execute(
        "SELECT data FROM metadata WHERE key = ?",
        (_learning_review_stamps_key(agent_id),),
    ).fetchone()
    if row is None:
        return ()
    value = _loads(row[0])
    if not isinstance(value, tuple) or any(
        not isinstance(item, LearningCandidateReviewStamp) for item in value
    ):
        raise RuntimeError("stored learning review stamps are invalid")
    if len(value) > LEARNING_REVIEW_MAX_STAMPS or len(set(value)) != len(value):
        raise RuntimeError("stored learning review stamps exceed their bound")
    return value


def _load_learning_candidate_required(
    connection: sqlite3.Connection,
    agent_id: str,
    candidate_id: str,
) -> LearningCandidate:
    row = connection.execute(
        """SELECT data FROM learning_candidates
           WHERE agent_id = ? AND id = ?""",
        (agent_id, candidate_id),
    ).fetchone()
    if row is None:
        raise LearningCandidateNotFoundError(candidate_id)
    candidate = _expect(_loads(row[0]), LearningCandidate)
    _validate_learning_candidate_owner(agent_id, candidate)
    return candidate


def _require_candidate_transition(
    current: LearningCandidate,
    *,
    expected_fingerprint: str,
) -> None:
    if current.status is not LearningCandidateStatus.AWAITING_REVIEW:
        raise LearningCandidateError(
            f"candidate is not awaiting review: {current.status.value}"
        )
    if current.candidate_fingerprint != expected_fingerprint:
        raise LearningCandidateError(
            "learning candidate changed; load it again before mutation"
        )


async def _run_candidate_transaction(
    path: Path,
    callback: Callable[[sqlite3.Connection], _T],
) -> _T:
    gate = _CatalogCommitGate()

    def write() -> _T | None:
        connection = _connect(path)
        try:
            if not gate.start(connection):
                return None
            result = callback(connection)
            connection.commit()
            return result
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

    worker = asyncio.create_task(asyncio.to_thread(write))
    cancelled_before_start = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled_before_start = (
                gate.cancel_before_start() or cancelled_before_start
            )
    result = worker.result()
    if cancelled_before_start:
        if result is not None:
            raise AssertionError("cancelled learning candidate transaction committed")
        raise asyncio.CancelledError
    if result is None:
        raise AssertionError(
            "learning candidate transaction stopped without cancellation"
        )
    return result


def _initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _validate_existing_state(path)
        return
    with _connect(path) as connection:
        connection.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS sources (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id)
            );
            CREATE TABLE IF NOT EXISTS syncs (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id)
            );
            CREATE TABLE IF NOT EXISTS snapshots (
                agent_id TEXT NOT NULL,
                source_id TEXT NOT NULL,
                sync_id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, source_id)
            );
            CREATE TABLE IF NOT EXISTS runs (
                id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                conversation_id TEXT NOT NULL,
                turn_index INTEGER NOT NULL,
                input TEXT NOT NULL,
                result TEXT
            );
            CREATE TABLE IF NOT EXISTS messages (
                run_id TEXT NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
                position INTEGER NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(run_id, position)
            );
            CREATE TABLE IF NOT EXISTS semantic_annotations (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id)
            );
            CREATE TABLE IF NOT EXISTS learning_candidates (
                agent_id TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL,
                PRIMARY KEY(agent_id, id)
            );
            """)
        connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS runs_conversation_turn
               ON runs(agent_id, conversation_id, turn_index)""")


def _validate_existing_state(path: Path) -> None:
    try:
        with _connect_read_only(path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master "
                    "WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if tables != set(_V1_TABLE_DEFINITIONS):
                raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE)
            for table, expected in _V1_TABLE_DEFINITIONS.items():
                actual = tuple(
                    (row[1], str(row[2]).upper(), row[3], row[4], row[5])
                    for row in connection.execute(f"PRAGMA table_info({table})")
                )
                if actual != expected:
                    raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE)

            messages_foreign_keys = tuple(
                (row[2], row[3], row[4], row[5], row[6], row[7])
                for row in connection.execute("PRAGMA foreign_key_list(messages)")
            )
            if messages_foreign_keys != _V1_MESSAGES_FOREIGN_KEYS:
                raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE)
            for table in set(_V1_TABLE_DEFINITIONS) - {"messages"}:
                if tuple(connection.execute(f"PRAGMA foreign_key_list({table})")):
                    raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE)

            named_indexes = {
                row[0]: row[1]
                for row in connection.execute(
                    "SELECT name, tbl_name FROM sqlite_master "
                    "WHERE type = 'index' AND name NOT LIKE 'sqlite_%'"
                )
            }
            if named_indexes != {
                name: definition[0] for name, definition in _V1_NAMED_INDEXES.items()
            }:
                raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE)
            for name, (
                table,
                expected_unique,
                expected_columns,
            ) in _V1_NAMED_INDEXES.items():
                indexes = {
                    row[1]: bool(row[2])
                    for row in connection.execute(f"PRAGMA index_list({table})")
                    if not str(row[1]).startswith("sqlite_autoindex")
                }
                if indexes != {name: expected_unique}:
                    raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE)
                columns = tuple(
                    row[2] for row in connection.execute(f"PRAGMA index_info({name})")
                )
                if columns != expected_columns:
                    raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE)

            extra_objects = tuple(
                connection.execute(
                    "SELECT type, name FROM sqlite_master "
                    "WHERE type IN ('trigger', 'view') "
                    "AND name NOT LIKE 'sqlite_%'"
                )
            )
            if extra_objects:
                raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE)
    except RuntimeError:
        raise
    except (OSError, sqlite3.Error):
        raise RuntimeError(_INCOMPATIBLE_STATE_MESSAGE) from None


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _connect_read_only(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(
        path.as_uri() + "?mode=ro",
        timeout=30,
        uri=True,
    )
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA query_only = ON")
    return connection


def _commit_catalog_transaction(connection: sqlite3.Connection) -> None:
    connection.commit()


_RECORD_TYPES: dict[str, type[Any]] = {
    record.__name__: record
    for record in (
        AgentIdentity,
        SourceRegistration,
        CatalogFacet,
        RelationshipFieldPair,
        CatalogRelationship,
        CatalogResourceRevision,
        CatalogResource,
        CatalogSync,
        SourceCatalogSnapshot,
        RunInput,
        LoopExit,
        TextBlock,
        ToolCall,
        ToolResultBlock,
        CanonicalMessage,
        ModelUsage,
        PricingModifier,
        PricingUsageRange,
        CostComponent,
        CostEstimate,
        SemanticFieldReference,
        SemanticEvidence,
        ResourceRevisionBinding,
        SemanticSubject,
        SemanticAnnotation,
        DocumentCandidateContent,
        SemanticCandidateContent,
        SkillCandidateContent,
        LearningCandidateRunReference,
        LearningCandidateReviewStamp,
        LearningCandidate,
    )
}
_ENUM_TYPES: dict[str, type[Enum]] = {
    enum.__name__: enum
    for enum in (
        CatalogSyncStatus,
        FacetKind,
        RelationshipDirection,
        RelationshipKind,
        RelationshipProvenance,
        ResourceKind,
        Sensitivity,
        LoopExitKind,
        MessageRole,
        CostBasis,
        CostEstimateStatus,
        SemanticEvidenceKind,
        SemanticKind,
        LearningCandidateStatus,
        LearningCandidateRejectionReason,
        LearningCandidateTarget,
        LearningCandidateAction,
    )
}


def _pack(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _pack(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__record__": type(value).__name__,
            "fields": {
                field.name: _pack(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, Enum):
        return {"__enum__": type(value).__name__, "value": value.value}
    if isinstance(value, datetime):
        return {"__datetime__": value.isoformat()}
    if isinstance(value, Decimal):
        return {"__decimal__": str(value)}
    if isinstance(value, (tuple, list)):
        return [_pack(item) for item in value]
    return value


def _unpack(value: object) -> object:
    if isinstance(value, list):
        return tuple(_unpack(item) for item in value)
    if not isinstance(value, dict):
        return value
    if "__datetime__" in value:
        return datetime.fromisoformat(str(value["__datetime__"]))
    if "__decimal__" in value:
        return Decimal(str(value["__decimal__"]))
    if "__enum__" in value:
        enum = _ENUM_TYPES[str(value["__enum__"])]
        return enum(value["value"])
    if "__record__" in value:
        record = _RECORD_TYPES[str(value["__record__"])]
        raw_fields = value["fields"]
        if not isinstance(raw_fields, dict):
            raise ValueError("invalid stored record fields")
        return record(**{name: _unpack(item) for name, item in raw_fields.items()})
    return {key: _unpack(item) for key, item in value.items()}


def _dumps(value: object) -> str:
    return json.dumps(_pack(value), sort_keys=True, separators=(",", ":"))


def _loads(value: str) -> object:
    return _unpack(json.loads(value))


def _expect(value: object, expected: type[Any]) -> Any:
    if not isinstance(value, expected):
        raise TypeError(f"stored value is not {expected.__name__}")
    return value


__all__ = ["SQLiteStateStore"]
