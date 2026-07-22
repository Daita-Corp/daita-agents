"""Small SQLite state store for one embedded agent.

The database persists only product state the MVP actually uses: identity,
attached sources, current catalog snapshots, and exact run transcripts.
"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import re
import sqlite3
from typing import Any

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
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ModelUsage,
    TextBlock,
    ToolCall,
    ToolResultBlock,
)
from ..loop.models import ConversationRun, LoopExit, LoopExitKind, RunInput, Transcript

_SEARCH_TERM = re.compile(r"[A-Za-z0-9]+")


class SQLiteStateStore:
    """One deliberately non-versioned persistence boundary for embedded MVP use."""

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

    async def detach_source(
        self, agent_id: str, source_id: str, detached_at: datetime
    ) -> SourceRegistration:
        def write() -> SourceRegistration:
            with _connect(self.path) as connection:
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
                return detached

        return await asyncio.to_thread(write)

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
        self, snapshot: SourceCatalogSnapshot
    ) -> SourceCatalogSnapshot:
        def write() -> SourceCatalogSnapshot:
            sync = snapshot.sync
            with _connect(self.path) as connection:
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
                return snapshot

        return await asyncio.to_thread(write)

    async def load_sync(self, agent_id: str, sync_id: str) -> CatalogSync | None:
        def read() -> CatalogSync | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM syncs WHERE agent_id = ? AND id = ?",
                    (agent_id, sync_id),
                ).fetchone()
            return None if row is None else _expect(_loads(row[0]), CatalogSync)

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
        terms = tuple(term.lower() for term in _SEARCH_TERM.findall(request.query))
        matches: list[CatalogSearchHit] = []
        for resource in await self.list_resources(request.agent_id):
            if request.source_ids and resource.source_id not in request.source_ids:
                continue
            if request.resource_kinds and resource.kind not in request.resource_kinds:
                continue
            fields = {
                "name": resource.name.lower(),
                "native_identity": resource.native_identity.lower(),
                "external_uri": resource.external_uri.lower(),
            }
            if terms and not all(
                any(term in value for value in fields.values()) for term in terms
            ):
                continue
            matched_fields = tuple(
                name
                for name, value in fields.items()
                if any(term in value for term in terms)
            )
            exact = bool(terms) and any(
                request.query.lower() == value for value in fields.values()
            )
            prefix = bool(terms) and any(
                value.startswith(request.query.lower()) for value in fields.values()
            )
            reason = (
                "lexical_exact"
                if exact
                else "lexical_prefix" if prefix else "lexical_contains"
            )
            score = (
                100.0
                if exact
                else 50.0 if prefix else float(max(1, len(matched_fields)))
            )
            matches.append(
                CatalogSearchHit(
                    resource_id=resource.id,
                    source_id=resource.source_id,
                    kind=resource.kind,
                    name=resource.name,
                    revision=resource.current_revision,
                    sensitivity=resource.sensitivity,
                    score=score,
                    matched_fields=matched_fields,
                    match_reasons=(reason,),
                )
            )
        matches.sort(key=lambda item: (-item.score, item.name, item.resource_id))
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


def _initialize(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            """)
        run_columns = tuple(
            row[1] for row in connection.execute("PRAGMA table_info(runs)")
        )
        expected_run_columns = (
            "id",
            "agent_id",
            "conversation_id",
            "turn_index",
            "input",
            "result",
        )
        if run_columns != expected_run_columns:
            raise RuntimeError(
                "state database uses a pre-conversation MVP schema; "
                "create a fresh agent home"
            )
        connection.execute("""CREATE UNIQUE INDEX IF NOT EXISTS runs_conversation_turn
               ON runs(agent_id, conversation_id, turn_index)""")


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


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
