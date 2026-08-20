"""Small SQLite state store for one embedded agent.

The database persists only product state the MVP actually uses: identity,
attached sources, current catalog snapshots, exact run transcripts, and the
minimal receipt needed to classify an external database-write attempt.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sqlite3
import tempfile
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import TypeVar

from .._json import FrozenJsonObject
from ..adapters.mcp import MCP_MAX_BINDINGS_PER_AGENT, MCPServerBinding
from ..adapters.models import SourceRegistration
from ..artifacts.models import (
    ArtifactRef,
    artifact_ref_from_mapping,
)
from ..catalog.models import (
    CatalogFacet,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSnapshotRef,
    CatalogSummary,
    CatalogSync,
    FacetKind,
    RelationshipKind,
    SourceCatalogSnapshot,
)
from ..catalog.protocols import CatalogStoreError
from ..errors import StateCompatibilityCode, StateCompatibilityError
from ..identity import AgentIdentity, AgentIdentityConflictError
from ..learning_candidates import (
    LEARNING_CANDIDATE_MAX_RECORDS,
    LEARNING_REVIEW_MAX_PROPOSALS,
    LEARNING_REVIEW_MAX_STAMPS,
    LearningCandidate,
    LearningCandidateError,
    LearningCandidateNotFoundError,
    LearningCandidateRejectionReason,
    LearningCandidateReviewStamp,
    LearningCandidateStatus,
    LearningReviewRunTail,
)
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
    ToolResultBlock,
)
from ..loop.models import (
    ConversationRun,
    LoopExit,
    LoopExitKind,
    RunInput,
    Transcript,
    validate_completed_transcript,
)
from ..semantics import (
    SEMANTIC_MAX_ANNOTATIONS,
    SemanticAnnotation,
    SemanticDigestMismatchError,
    SemanticNotFoundError,
    SemanticValidationError,
    semantic_annotation_sha256,
)
from .sqlite_codecs import (
    decode_catalog_snapshot,
    decode_catalog_sync,
    decode_identifier,
    decode_identity,
    decode_learning_candidate,
    decode_loop_exit,
    decode_mcp_binding,
    decode_message,
    decode_postgresql_update_scope,
    decode_receipt,
    decode_review_stamps,
    decode_run_input,
    decode_semantic_annotation,
    decode_source,
    decode_source_read_scope,
    encode_catalog_snapshot,
    encode_catalog_sync,
    encode_identifier,
    encode_identity,
    encode_learning_candidate,
    encode_loop_exit,
    encode_mcp_binding,
    encode_message,
    encode_postgresql_update_scope,
    encode_receipt,
    encode_review_stamps,
    encode_run_input,
    encode_semantic_annotation,
    encode_source,
    encode_source_read_scope,
)
from .sqlite_migrations import (
    CURRENT_REVISION,
    MIGRATIONS,
    MigrationJournalError,
    MigrationJournalNewerError,
    create_current,
    inspect_journal,
    upgrade_journaled,
)
from .sqlite_migrations.preledger import (
    PreledgerAdmissionError,
    PreledgerLegacyError,
    PreledgerNewerError,
    PreledgerShape,
    bridge as bridge_preledger,
    identify as identify_preledger,
)
from .sqlite_records import (
    DatabaseWriteOutcome,
    DatabaseWriteReceipt,
    DatabaseWriteReceiptConflictError,
    PostgreSQLUpdateScope,
    SourcePermissionStateError,
    SourceReadMode,
    SourceReadScope,
    database_write_aware as _database_write_aware,
    database_write_receipt_id,
    database_write_text as _database_write_text,
    postgresql_update_authorization_fingerprint,
    validate_database_write_receipt_id,
)
from .sqlite_schema import (
    CURRENT_TABLES,
    require_healthy,
    require_schema,
    table_names,
)

_CATALOG_SNAPSHOT_SOURCE_FILTER_BATCH = 64
_ACTIVE_SOURCE_KEY_PREFIX = "active_source:"
_LEARNING_REVIEW_STAMPS_KEY_PREFIX = "learning_review_stamps:"
_T = TypeVar("_T")


def _active_source_key(agent_id: str) -> str:
    return f"{_ACTIVE_SOURCE_KEY_PREFIX}{agent_id}"


def _learning_review_stamps_key(agent_id: str) -> str:
    return f"{_LEARNING_REVIEW_STAMPS_KEY_PREFIX}{agent_id}"


def _decode_owned_read_scope(
    connection: sqlite3.Connection,
    registration: SourceRegistration,
    data: object,
) -> SourceReadScope | None:
    if not registration.active:
        if data is not None:
            raise SourcePermissionStateError(
                "detached source unexpectedly retains a read scope"
            )
        return None
    if not isinstance(data, str):
        raise SourcePermissionStateError("active source is missing its read scope")
    try:
        scope = decode_source_read_scope(
            data,
            agent_id=registration.agent_id,
            source_id=registration.id,
        )
        if scope.mode is SourceReadMode.SELECTED:
            _require_nonforeign_read_resources(connection, scope)
        return scope
    except SourcePermissionStateError:
        raise
    except (TypeError, ValueError):
        raise SourcePermissionStateError(
            "active source read scope is undecodable"
        ) from None


def _require_nonforeign_read_resources(
    connection: sqlite3.Connection,
    scope: SourceReadScope,
) -> None:
    requested = set(scope.resource_ids)
    if not requested:
        return
    for (data,) in connection.execute(
        "SELECT data FROM snapshots WHERE agent_id = ?",
        (scope.agent_id,),
    ):
        snapshot = decode_catalog_snapshot(data)
        for resource in snapshot.resources:
            if resource.id in requested and resource.source_id != scope.source_id:
                raise SourcePermissionStateError(
                    "active source read scope contains a foreign resource"
                )


def _decode_source_state(
    connection: sqlite3.Connection,
    *,
    row_agent_id: object,
    row_source_id: object,
    source_data: object,
    read_scope_data: object,
    update_scope_count: object,
) -> SourceRegistration:
    if (
        not isinstance(row_agent_id, str)
        or not isinstance(row_source_id, str)
        or not isinstance(source_data, str)
        or not isinstance(update_scope_count, int)
        or isinstance(update_scope_count, bool)
        or update_scope_count < 0
    ):
        raise SourcePermissionStateError("stored source permission state is invalid")
    try:
        registration = decode_source(source_data)
    except (TypeError, ValueError):
        raise SourcePermissionStateError(
            "stored source registration is invalid"
        ) from None
    if registration.agent_id != row_agent_id or registration.id != row_source_id:
        raise SourcePermissionStateError("stored source ownership is invalid")
    _decode_owned_read_scope(connection, registration, read_scope_data)
    if update_scope_count and (
        not registration.active or registration.adapter_id != "postgresql"
    ):
        raise SourcePermissionStateError("stored PostgreSQL update scope is foreign")
    return registration


def _source_state_row(
    connection: sqlite3.Connection,
    agent_id: str,
    source_id: str,
) -> tuple[object, ...] | None:
    return connection.execute(
        """SELECT s.agent_id, s.id, s.data, r.data,
                  (SELECT COUNT(*) FROM postgresql_update_scopes AS u
                   WHERE u.agent_id = s.agent_id AND u.source_id = s.id)
           FROM sources AS s
           LEFT JOIN source_read_scopes AS r
             ON r.agent_id = s.agent_id AND r.source_id = s.id
           WHERE s.agent_id = ? AND s.id = ?""",
        (agent_id, source_id),
    ).fetchone()


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


class _UpgradeCommitGate:
    """Let task cancellation veto a not-yet-committed state migration."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._committed = False

    def start(self, connection: sqlite3.Connection) -> bool:
        with self._lock:
            if self._cancelled:
                return False
        connection.execute("BEGIN IMMEDIATE")
        with self._lock:
            if self._cancelled:
                connection.rollback()
                return False
            return True

    def commit(self, connection: sqlite3.Connection) -> bool:
        with self._lock:
            if self._cancelled:
                connection.rollback()
                return False
            connection.commit()
            return True

    def activate(self, callback: Callable[[], None]) -> bool:
        with self._lock:
            if self._cancelled:
                return False
            callback()
            self._committed = True
            return True

    def cancel_before_commit(self) -> bool:
        with self._lock:
            if self._committed:
                return False
            self._cancelled = True
            return True


class SQLiteStateStore:
    """Sole persistence and upgrade boundary for one admitted agent home."""

    current_revision = CURRENT_REVISION

    def __init__(self, path: Path) -> None:
        self.path = path
        self._decoded_catalog_snapshots: dict[
            tuple[str, str, str], SourceCatalogSnapshot
        ] = {}
        self._decoded_catalog_snapshot_lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        path: str | Path,
        *,
        clock: Callable[[], datetime] | None = None,
        **_: object,
    ) -> SQLiteStateStore:
        resolved = Path(path).resolve()
        resolved_clock = clock or (lambda: datetime.now(UTC))

        upgrade_gate = _UpgradeCommitGate()

        def admit() -> None:
            if _initialize(resolved, upgrade_gate=upgrade_gate):
                _recover_started_database_write_receipts(resolved, resolved_clock)

        worker = asyncio.create_task(asyncio.to_thread(admit))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
                upgrade_gate.cancel_before_commit()
        worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return cls(resolved)

    async def close(self) -> None:
        async with self._decoded_catalog_snapshot_lock:
            self._decoded_catalog_snapshots.clear()

    async def initialize_identity(self, identity: AgentIdentity) -> AgentIdentity:
        def write() -> AgentIdentity:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
                if row is not None:
                    current = decode_identity(row[0])
                    if current != identity:
                        raise AgentIdentityConflictError(
                            "state database already belongs to another agent"
                        )
                    return current
                connection.execute(
                    "INSERT INTO metadata(key, data) VALUES ('identity', ?)",
                    (encode_identity(identity),),
                )
                return identity

        return await asyncio.to_thread(write)

    async def load_identity(self) -> AgentIdentity | None:
        def read() -> AgentIdentity | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
            return None if row is None else decode_identity(row[0])

        return await asyncio.to_thread(read)

    async def load_mcp_binding(
        self,
        agent_id: str,
        binding_id: str,
    ) -> MCPServerBinding | None:
        def read() -> MCPServerBinding | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM mcp_server_bindings
                       WHERE agent_id = ? AND binding_id = ?""",
                    (agent_id, binding_id),
                ).fetchone()
            return (
                None
                if row is None
                else decode_mcp_binding(
                    row[0],
                    agent_id=agent_id,
                    binding_id=binding_id,
                )
            )

        return await asyncio.to_thread(read)

    async def list_mcp_bindings(
        self,
        agent_id: str,
    ) -> tuple[MCPServerBinding, ...]:
        def read() -> tuple[MCPServerBinding, ...]:
            with _connect(self.path) as connection:
                rows = tuple(
                    connection.execute(
                        """SELECT binding_id, data FROM mcp_server_bindings
                           WHERE agent_id = ? ORDER BY binding_id""",
                        (agent_id,),
                    )
                )
            if len(rows) > MCP_MAX_BINDINGS_PER_AGENT:
                raise RuntimeError("stored MCP binding count exceeds its fixed bound")
            return tuple(
                decode_mcp_binding(
                    data,
                    agent_id=agent_id,
                    binding_id=binding_id,
                )
                for binding_id, data in rows
            )

        return await asyncio.to_thread(read)

    async def store_mcp_binding(
        self,
        binding: MCPServerBinding,
        *,
        expected_revision: int | None,
    ) -> MCPServerBinding:
        if not isinstance(binding, MCPServerBinding):
            raise TypeError("binding must be MCPServerBinding")
        if expected_revision is not None and (
            not isinstance(expected_revision, int)
            or isinstance(expected_revision, bool)
            or expected_revision < 1
        ):
            raise ValueError("expected_revision must be positive or None")

        def write(connection: sqlite3.Connection) -> MCPServerBinding:
            row = connection.execute(
                """SELECT data FROM mcp_server_bindings
                   WHERE agent_id = ? AND binding_id = ?""",
                (binding.agent_id, binding.binding_id),
            ).fetchone()
            if row is None:
                if expected_revision is not None or binding.revision != 1:
                    raise ValueError("MCP binding revision precondition failed")
                count = connection.execute(
                    "SELECT COUNT(*) FROM mcp_server_bindings WHERE agent_id = ?",
                    (binding.agent_id,),
                ).fetchone()[0]
                if count >= MCP_MAX_BINDINGS_PER_AGENT:
                    raise ValueError("MCP binding count exceeds its fixed bound")
                connection.execute(
                    """INSERT INTO mcp_server_bindings(agent_id, binding_id, data)
                       VALUES (?, ?, ?)""",
                    (
                        binding.agent_id,
                        binding.binding_id,
                        encode_mcp_binding(binding),
                    ),
                )
                return binding
            current = decode_mcp_binding(
                row[0],
                agent_id=binding.agent_id,
                binding_id=binding.binding_id,
            )
            if (
                expected_revision is None
                or current.revision != expected_revision
                or binding.revision != expected_revision + 1
            ):
                raise ValueError("MCP binding revision precondition failed")
            result = connection.execute(
                """UPDATE mcp_server_bindings SET data = ?
                   WHERE agent_id = ? AND binding_id = ? AND data = ?""",
                (
                    encode_mcp_binding(binding),
                    binding.agent_id,
                    binding.binding_id,
                    row[0],
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("MCP binding changed during its transition")
            return binding

        return await _run_cancellation_safe_transaction(self.path, write)

    async def register_source(
        self, registration: SourceRegistration
    ) -> SourceRegistration:
        def write() -> SourceRegistration:
            stored = registration
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = _source_state_row(
                    connection,
                    registration.agent_id,
                    registration.id,
                )
                if row is not None:
                    current = _decode_source_state(
                        connection,
                        row_agent_id=row[0],
                        row_source_id=row[1],
                        source_data=row[2],
                        read_scope_data=row[3],
                        update_scope_count=row[4],
                    )
                    if current != stored:
                        raise ValueError(
                            f"source registration already exists: {registration.id}"
                        )
                    return current
                connection.execute(
                    "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                    (stored.agent_id, stored.id, encode_source(stored)),
                )
                if stored.active:
                    scope = SourceReadScope.allow_all(
                        agent_id=stored.agent_id,
                        source_id=stored.id,
                    )
                    connection.execute(
                        """INSERT INTO source_read_scopes(agent_id, source_id, data)
                           VALUES (?, ?, ?)""",
                        (stored.agent_id, stored.id, encode_source_read_scope(scope)),
                    )
                return stored

        return await asyncio.to_thread(write)

    async def load_source(
        self, agent_id: str, source_id: str
    ) -> SourceRegistration | None:
        def read() -> SourceRegistration | None:
            with _connect(self.path) as connection:
                row = _source_state_row(connection, agent_id, source_id)
                return (
                    None
                    if row is None
                    else _decode_source_state(
                        connection,
                        row_agent_id=row[0],
                        row_source_id=row[1],
                        source_data=row[2],
                        read_scope_data=row[3],
                        update_scope_count=row[4],
                    )
                )

        return await asyncio.to_thread(read)

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        def read() -> tuple[SourceRegistration, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT s.agent_id, s.id, s.data, r.data,
                              (SELECT COUNT(*)
                               FROM postgresql_update_scopes AS u
                               WHERE u.agent_id = s.agent_id
                                 AND u.source_id = s.id)
                       FROM sources AS s
                       LEFT JOIN source_read_scopes AS r
                         ON r.agent_id = s.agent_id AND r.source_id = s.id
                       WHERE s.agent_id = ? ORDER BY s.id""",
                    (agent_id,),
                ).fetchall()
                return tuple(
                    _decode_source_state(
                        connection,
                        row_agent_id=row_agent_id,
                        row_source_id=row_source_id,
                        source_data=data,
                        read_scope_data=read_scope_data,
                        update_scope_count=update_scope_count,
                    )
                    for (
                        row_agent_id,
                        row_source_id,
                        data,
                        read_scope_data,
                        update_scope_count,
                    ) in rows
                )

        return await asyncio.to_thread(read)

    async def load_source_read_scope(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceReadScope | None:
        """Load one exact active-source scope; invalid state never becomes all."""

        def read() -> SourceReadScope | None:
            with _connect(self.path) as connection:
                row = _source_state_row(connection, agent_id, source_id)
                if row is None:
                    return None
                registration = _decode_source_state(
                    connection,
                    row_agent_id=row[0],
                    row_source_id=row[1],
                    source_data=row[2],
                    read_scope_data=row[3],
                    update_scope_count=row[4],
                )
                return _decode_owned_read_scope(connection, registration, row[3])

        return await asyncio.to_thread(read)

    async def list_postgresql_update_scopes(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[PostgreSQLUpdateScope, ...]:
        def read() -> tuple[PostgreSQLUpdateScope, ...]:
            with _connect(self.path) as connection:
                source_row = _source_state_row(connection, agent_id, source_id)
                if source_row is None:
                    return ()
                registration = _decode_source_state(
                    connection,
                    row_agent_id=source_row[0],
                    row_source_id=source_row[1],
                    source_data=source_row[2],
                    read_scope_data=source_row[3],
                    update_scope_count=source_row[4],
                )
                if not registration.active:
                    return ()
                if registration.adapter_id != "postgresql":
                    if source_row[4]:
                        raise SourcePermissionStateError(
                            "non-PostgreSQL source retains update scopes"
                        )
                    return ()
                rows = connection.execute(
                    """SELECT resource_id, authorization_fingerprint, data
                       FROM postgresql_update_scopes
                       WHERE agent_id = ? AND source_id = ?
                       ORDER BY resource_id""",
                    (agent_id, source_id),
                ).fetchall()
                try:
                    return tuple(
                        decode_postgresql_update_scope(
                            data,
                            agent_id=agent_id,
                            source_id=source_id,
                            resource_id=resource_id,
                            authorization_fingerprint=fingerprint,
                        )
                        for resource_id, fingerprint, data in rows
                    )
                except (TypeError, ValueError):
                    raise SourcePermissionStateError(
                        "stored PostgreSQL update scope is undecodable"
                    ) from None

        return await asyncio.to_thread(read)

    async def replace_source_permission_scopes(
        self,
        read_scope: SourceReadScope,
        update_scopes: tuple[PostgreSQLUpdateScope, ...],
    ) -> SourceRegistration:
        """Atomically replace only the two narrow scope families for one source."""

        if not isinstance(read_scope, SourceReadScope):
            raise TypeError("read_scope must be a SourceReadScope")
        if not isinstance(update_scopes, tuple) or any(
            not isinstance(scope, PostgreSQLUpdateScope) for scope in update_scopes
        ):
            raise TypeError("update_scopes must be a tuple of PostgreSQLUpdateScope")
        if len({scope.resource_id for scope in update_scopes}) != len(update_scopes):
            raise ValueError("update_scopes cannot contain duplicate resources")
        gate = _CatalogCommitGate()

        def write() -> SourceRegistration | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                row = _source_state_row(
                    connection,
                    read_scope.agent_id,
                    read_scope.source_id,
                )
                if row is None:
                    raise ValueError("permission scopes require an active owned source")
                registration = _decode_source_state(
                    connection,
                    row_agent_id=row[0],
                    row_source_id=row[1],
                    source_data=row[2],
                    read_scope_data=row[3],
                    update_scope_count=row[4],
                )
                if not registration.active:
                    raise ValueError("permission scopes require an active owned source")
                _require_nonforeign_read_resources(connection, read_scope)
                if any(
                    scope.agent_id != read_scope.agent_id
                    or scope.source_id != read_scope.source_id
                    for scope in update_scopes
                ):
                    raise ValueError("update scope belongs to another source")
                update_resource_ids = {scope.resource_id for scope in update_scopes}
                if read_scope.mode is SourceReadMode.NONE and update_resource_ids:
                    raise ValueError("PostgreSQL update scope requires read access")
                if (
                    read_scope.mode is SourceReadMode.SELECTED
                    and not update_resource_ids <= set(read_scope.resource_ids)
                ):
                    raise ValueError("PostgreSQL update scope must be a read subset")
                if update_scopes and registration.adapter_id != "postgresql":
                    raise ValueError(
                        "PostgreSQL update scopes require a PostgreSQL source"
                    )

                snapshot = _current_source_snapshot(
                    connection,
                    read_scope.agent_id,
                    read_scope.source_id,
                )
                resources = (
                    {}
                    if snapshot is None
                    else {item.id: item for item in snapshot.resources}
                )
                facets = (
                    {}
                    if snapshot is None
                    else {
                        item.resource_id: item
                        for item in snapshot.facets
                        if item.kind is FacetKind.TABULAR
                    }
                )
                for scope in update_scopes:
                    resource = resources.get(scope.resource_id)
                    facet = facets.get(scope.resource_id)
                    if resource is None or facet is None:
                        raise ValueError(
                            "PostgreSQL update scope requires a current table resource"
                        )
                    expected = postgresql_update_authorization_fingerprint(
                        source=registration,
                        resource=resource,
                        facet=facet,
                        allowed_assignment_columns=scope.allowed_assignment_columns,
                    )
                    if scope.authorization_fingerprint != expected:
                        raise ValueError(
                            "PostgreSQL update authorization fingerprint is stale"
                        )

                connection.execute(
                    """INSERT INTO source_read_scopes(agent_id, source_id, data)
                       VALUES (?, ?, ?)
                       ON CONFLICT(agent_id, source_id)
                       DO UPDATE SET data = excluded.data""",
                    (
                        read_scope.agent_id,
                        read_scope.source_id,
                        encode_source_read_scope(read_scope),
                    ),
                )
                connection.execute(
                    """DELETE FROM postgresql_update_scopes
                       WHERE agent_id = ? AND source_id = ?""",
                    (read_scope.agent_id, read_scope.source_id),
                )
                for scope in sorted(update_scopes, key=lambda item: item.resource_id):
                    connection.execute(
                        """INSERT INTO postgresql_update_scopes(
                               agent_id, source_id, resource_id,
                               authorization_fingerprint, data
                           ) VALUES (?, ?, ?, ?, ?)""",
                        (
                            scope.agent_id,
                            scope.source_id,
                            scope.resource_id,
                            scope.authorization_fingerprint,
                            encode_postgresql_update_scope(scope),
                        ),
                    )
                connection.commit()
                return registration
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
        updated = worker.result()
        if cancelled_before_start:
            if updated is not None:
                raise AssertionError(
                    "cancelled source permission transaction committed"
                )
            raise asyncio.CancelledError
        if updated is None:
            raise AssertionError(
                "source permission transaction stopped without cancellation"
            )
        return updated

    async def load_database_write_receipt(
        self,
        agent_id: str,
        receipt_id: str,
    ) -> DatabaseWriteReceipt | None:
        _database_write_text(agent_id, "receipt agent_id")
        validate_database_write_receipt_id(receipt_id)

        def read() -> DatabaseWriteReceipt | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, receipt_id),
                ).fetchone()
            return None if row is None else decode_receipt(row[0])

        return await asyncio.to_thread(read)

    async def load_database_write_receipt_for_call(
        self,
        agent_id: str,
        run_id: str,
        call_id: str,
    ) -> DatabaseWriteReceipt | None:
        _database_write_text(agent_id, "receipt agent_id")
        _database_write_text(run_id, "receipt run_id")
        _database_write_text(call_id, "receipt call_id")

        def read() -> DatabaseWriteReceipt | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ? AND run_id = ? AND call_id = ?""",
                    (agent_id, run_id, call_id),
                ).fetchone()
            return None if row is None else decode_receipt(row[0])

        return await asyncio.to_thread(read)

    async def start_database_write_receipt(
        self,
        receipt: DatabaseWriteReceipt,
    ) -> DatabaseWriteReceipt:
        if not isinstance(receipt, DatabaseWriteReceipt):
            raise TypeError("receipt must be a DatabaseWriteReceipt")
        if receipt.outcome is not DatabaseWriteOutcome.STARTED:
            raise ValueError(
                "database write execution must begin with a started receipt"
            )

        def write() -> DatabaseWriteReceipt:
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                current = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ?
                         AND (id = ? OR (run_id = ? AND call_id = ?))""",
                    (
                        receipt.agent_id,
                        receipt.receipt_id,
                        receipt.run_id,
                        receipt.call_id,
                    ),
                ).fetchone()
                if current is not None:
                    raise DatabaseWriteReceiptConflictError(
                        "execution identity already has a receipt"
                    )
                connection.execute(
                    """INSERT INTO database_write_receipts(
                           agent_id, id, run_id, call_id, data
                       ) VALUES (?, ?, ?, ?, ?)""",
                    (
                        receipt.agent_id,
                        receipt.receipt_id,
                        receipt.run_id,
                        receipt.call_id,
                        encode_receipt(receipt),
                    ),
                )
                return receipt

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        result = worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def finish_database_write_receipt(
        self,
        receipt: DatabaseWriteReceipt,
    ) -> DatabaseWriteReceipt:
        if not isinstance(receipt, DatabaseWriteReceipt):
            raise TypeError("receipt must be a DatabaseWriteReceipt")
        if receipt.outcome is DatabaseWriteOutcome.STARTED:
            raise ValueError(
                "finish requires a terminal receipt, not a started receipt"
            )

        def write() -> DatabaseWriteReceipt:
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                row = connection.execute(
                    """SELECT data FROM database_write_receipts
                       WHERE agent_id = ?
                         AND (id = ? OR (run_id = ? AND call_id = ?))""",
                    (
                        receipt.agent_id,
                        receipt.receipt_id,
                        receipt.run_id,
                        receipt.call_id,
                    ),
                ).fetchone()
                if row is None:
                    raise DatabaseWriteReceiptConflictError(
                        "started receipt does not exist"
                    )
                current = decode_receipt(row[0])
                if current.outcome is not DatabaseWriteOutcome.STARTED:
                    if current == receipt:
                        return current
                    raise DatabaseWriteReceiptConflictError(
                        "terminal receipt is immutable"
                    )
                if current != receipt.as_started():
                    raise DatabaseWriteReceiptConflictError(
                        "terminal receipt does not match its started identity"
                    )
                connection.execute(
                    """UPDATE database_write_receipts SET data = ?
                       WHERE agent_id = ? AND id = ?""",
                    (encode_receipt(receipt), receipt.agent_id, receipt.receipt_id),
                )
                return receipt

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        result = worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def load_active_source_id(self, agent_id: str) -> str | None:
        def read() -> str | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = ?",
                    (_active_source_key(agent_id),),
                ).fetchone()
            if row is None:
                return None
            source_id = decode_identifier(row[0])
            if not source_id:
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
                row = _source_state_row(connection, agent_id, source_id)
                if row is None:
                    raise ValueError("unknown active source for this agent")
                registration = _decode_source_state(
                    connection,
                    row_agent_id=row[0],
                    row_source_id=row[1],
                    source_data=row[2],
                    read_scope_data=row[3],
                    update_scope_count=row[4],
                )
                if not registration.active:
                    raise ValueError("unknown active source for this agent")
                connection.execute(
                    """INSERT INTO metadata(key, data) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                    (_active_source_key(agent_id), encode_identifier(source_id)),
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
                current = decode_source(row[0])
                detached = (
                    current if not current.active else current.detach(detached_at)
                )
                connection.execute(
                    "UPDATE sources SET data = ? WHERE agent_id = ? AND id = ?",
                    (encode_source(detached), agent_id, source_id),
                )
                connection.execute(
                    """DELETE FROM source_read_scopes
                       WHERE agent_id = ? AND source_id = ?""",
                    (agent_id, source_id),
                )
                connection.execute(
                    """DELETE FROM postgresql_update_scopes
                       WHERE agent_id = ? AND source_id = ?""",
                    (agent_id, source_id),
                )
                selection = connection.execute(
                    "SELECT data FROM metadata WHERE key = ?",
                    (_active_source_key(agent_id),),
                ).fetchone()
                selected_id = (
                    None if selection is None else decode_identifier(selection[0])
                )
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
                        candidate = decode_source(data)
                        if candidate.active:
                            remaining.append(candidate)
                    if len(remaining) == 1:
                        connection.execute(
                            "INSERT INTO metadata(key, data) VALUES (?, ?)",
                            (
                                _active_source_key(agent_id),
                                encode_identifier(remaining[0].id),
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
        async with self._decoded_catalog_snapshot_lock:
            self._evict_decoded_catalog_source(agent_id, source_id)
        return detached

    async def record_sync(self, sync: CatalogSync) -> CatalogSync:
        def write() -> CatalogSync:
            with _connect(self.path) as connection:
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.id,
                        sync.source_id,
                        encode_catalog_sync(sync),
                    ),
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
                    stored_registration = registration
                    row = _source_state_row(
                        connection,
                        stored_registration.agent_id,
                        stored_registration.id,
                    )
                    if row is None:
                        connection.execute(
                            "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                            (
                                stored_registration.agent_id,
                                stored_registration.id,
                                encode_source(stored_registration),
                            ),
                        )
                        if stored_registration.active:
                            attach_scope = SourceReadScope.allow_all(
                                agent_id=stored_registration.agent_id,
                                source_id=stored_registration.id,
                            )
                            connection.execute(
                                """INSERT INTO source_read_scopes(
                                       agent_id, source_id, data
                                   ) VALUES (?, ?, ?)""",
                                (
                                    stored_registration.agent_id,
                                    stored_registration.id,
                                    encode_source_read_scope(attach_scope),
                                ),
                            )
                    else:
                        current = _decode_source_state(
                            connection,
                            row_agent_id=row[0],
                            row_source_id=row[1],
                            source_data=row[2],
                            read_scope_data=row[3],
                            update_scope_count=row[4],
                        )
                        if current != stored_registration:
                            if (
                                current.active
                                or not stored_registration.active
                                or current.adapter_id != stored_registration.adapter_id
                                or current.native_identity
                                != stored_registration.native_identity
                            ):
                                raise ValueError(
                                    "source registration already exists: "
                                    f"{stored_registration.id}"
                                )
                            connection.execute(
                                """UPDATE sources SET data = ?
                                   WHERE agent_id = ? AND id = ?""",
                                (
                                    encode_source(stored_registration),
                                    stored_registration.agent_id,
                                    stored_registration.id,
                                ),
                            )
                            attach_scope = SourceReadScope.allow_all(
                                agent_id=stored_registration.agent_id,
                                source_id=stored_registration.id,
                            )
                            connection.execute(
                                """INSERT INTO source_read_scopes(
                                       agent_id, source_id, data
                                   ) VALUES (?, ?, ?)""",
                                (
                                    stored_registration.agent_id,
                                    stored_registration.id,
                                    encode_source_read_scope(attach_scope),
                                ),
                            )
                    selection = connection.execute(
                        "SELECT 1 FROM metadata WHERE key = ?",
                        (_active_source_key(stored_registration.agent_id),),
                    ).fetchone()
                    if selection is None:
                        connection.execute(
                            "INSERT INTO metadata(key, data) VALUES (?, ?)",
                            (
                                _active_source_key(stored_registration.agent_id),
                                encode_identifier(stored_registration.id),
                            ),
                        )
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.id,
                        sync.source_id,
                        encode_catalog_sync(sync),
                    ),
                )
                connection.execute(
                    """INSERT INTO snapshots(agent_id, source_id, sync_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, source_id) DO UPDATE SET
                         sync_id = excluded.sync_id, data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.source_id,
                        sync.id,
                        encode_catalog_snapshot(snapshot),
                    ),
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
        async with self._decoded_catalog_snapshot_lock:
            self._publish_decoded_catalog_snapshot(committed)
        return committed

    async def commit_source_edit(
        self,
        snapshot: SourceCatalogSnapshot,
        *,
        registration: SourceRegistration,
        replaced_source_id: str,
        replaced_at: datetime,
        read_scope: SourceReadScope,
    ) -> SourceCatalogSnapshot:
        """Atomically hand one active source connection to discovered truth."""

        if not isinstance(registration, SourceRegistration) or not registration.active:
            raise ValueError("source edit requires an active replacement registration")
        if not isinstance(replaced_source_id, str) or not replaced_source_id:
            raise ValueError("replaced_source_id must be a non-empty string")
        sync = snapshot.sync
        if registration.agent_id != sync.agent_id or registration.id != sync.source_id:
            raise ValueError("catalog snapshot and replacement registration disagree")
        if (
            read_scope.agent_id != registration.agent_id
            or read_scope.source_id != registration.id
        ):
            raise ValueError("replacement read scope belongs to another source")
        snapshot_resource_ids = {resource.id for resource in snapshot.resources}
        if (
            read_scope.mode is SourceReadMode.SELECTED
            and not set(read_scope.resource_ids) <= snapshot_resource_ids
        ):
            raise ValueError("replacement read scope is outside discovered catalog")
        gate = _CatalogCommitGate()

        def write() -> SourceCatalogSnapshot | None:
            connection = _connect(self.path)
            try:
                if not gate.start(connection):
                    return None
                replaced_row = _source_state_row(
                    connection,
                    registration.agent_id,
                    replaced_source_id,
                )
                if replaced_row is None:
                    raise ValueError("source edit requires an active owned source")
                replaced = _decode_source_state(
                    connection,
                    row_agent_id=replaced_row[0],
                    row_source_id=replaced_row[1],
                    source_data=replaced_row[2],
                    read_scope_data=replaced_row[3],
                    update_scope_count=replaced_row[4],
                )
                if not replaced.active:
                    raise ValueError("source edit requires an active owned source")
                if replaced.adapter_id != registration.adapter_id:
                    raise ValueError("source edit cannot change source type")

                if registration.id != replaced.id:
                    replacement_row = _source_state_row(
                        connection,
                        registration.agent_id,
                        registration.id,
                    )
                    if replacement_row is None:
                        connection.execute(
                            "INSERT INTO sources(agent_id, id, data) VALUES (?, ?, ?)",
                            (
                                registration.agent_id,
                                registration.id,
                                encode_source(registration),
                            ),
                        )
                    else:
                        existing_replacement = _decode_source_state(
                            connection,
                            row_agent_id=replacement_row[0],
                            row_source_id=replacement_row[1],
                            source_data=replacement_row[2],
                            read_scope_data=replacement_row[3],
                            update_scope_count=replacement_row[4],
                        )
                        if existing_replacement.active:
                            raise ValueError(
                                "replacement connection is already attached"
                            )
                        if (
                            existing_replacement.adapter_id != registration.adapter_id
                            or existing_replacement.native_identity
                            != registration.native_identity
                        ):
                            raise ValueError(
                                "replacement source identity conflicts with stored state"
                            )
                        connection.execute(
                            """UPDATE sources SET data = ?
                               WHERE agent_id = ? AND id = ?""",
                            (
                                encode_source(registration),
                                registration.agent_id,
                                registration.id,
                            ),
                        )
                    detached = replaced.detach(replaced_at)
                    connection.execute(
                        """UPDATE sources SET data = ?
                           WHERE agent_id = ? AND id = ?""",
                        (
                            encode_source(detached),
                            replaced.agent_id,
                            replaced.id,
                        ),
                    )
                else:
                    connection.execute(
                        """UPDATE sources SET data = ?
                           WHERE agent_id = ? AND id = ?""",
                        (
                            encode_source(registration),
                            registration.agent_id,
                            registration.id,
                        ),
                    )

                for source_id in {replaced.id, registration.id}:
                    connection.execute(
                        """DELETE FROM source_read_scopes
                           WHERE agent_id = ? AND source_id = ?""",
                        (registration.agent_id, source_id),
                    )
                    connection.execute(
                        """DELETE FROM postgresql_update_scopes
                           WHERE agent_id = ? AND source_id = ?""",
                        (registration.agent_id, source_id),
                    )
                connection.execute(
                    """INSERT INTO source_read_scopes(agent_id, source_id, data)
                       VALUES (?, ?, ?)""",
                    (
                        read_scope.agent_id,
                        read_scope.source_id,
                        encode_source_read_scope(read_scope),
                    ),
                )
                connection.execute(
                    """INSERT INTO metadata(key, data) VALUES (?, ?)
                       ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                    (
                        _active_source_key(registration.agent_id),
                        encode_identifier(registration.id),
                    ),
                )
                connection.execute(
                    """INSERT INTO syncs(agent_id, id, source_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.id,
                        sync.source_id,
                        encode_catalog_sync(sync),
                    ),
                )
                connection.execute(
                    """INSERT INTO snapshots(agent_id, source_id, sync_id, data)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(agent_id, source_id) DO UPDATE SET
                         sync_id = excluded.sync_id, data = excluded.data""",
                    (
                        sync.agent_id,
                        sync.source_id,
                        sync.id,
                        encode_catalog_snapshot(snapshot),
                    ),
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
                raise AssertionError("cancelled source edit transaction committed")
            raise asyncio.CancelledError
        if committed is None:
            raise AssertionError("source edit transaction stopped without cancellation")
        async with self._decoded_catalog_snapshot_lock:
            self._evict_decoded_catalog_source(
                registration.agent_id,
                replaced_source_id,
            )
            self._publish_decoded_catalog_snapshot(committed)
        return committed

    async def list_current_snapshot_refs(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
    ) -> tuple[CatalogSnapshotRef, ...]:
        _catalog_identifier(agent_id, "catalog snapshot agent_id")
        selected_source_ids = _catalog_snapshot_source_ids(source_ids)

        def read() -> tuple[CatalogSnapshotRef, ...]:
            with _connect(self.path) as connection:
                if not selected_source_ids:
                    rows = connection.execute(
                        "SELECT agent_id, source_id, sync_id FROM snapshots "
                        "WHERE agent_id = ? ORDER BY source_id, sync_id",
                        (agent_id,),
                    ).fetchall()
                else:
                    rows = []
                    for offset in range(
                        0,
                        len(selected_source_ids),
                        _CATALOG_SNAPSHOT_SOURCE_FILTER_BATCH,
                    ):
                        batch = selected_source_ids[
                            offset : offset + _CATALOG_SNAPSHOT_SOURCE_FILTER_BATCH
                        ]
                        placeholders = ", ".join("?" for _ in batch)
                        rows.extend(
                            connection.execute(
                                "SELECT agent_id, source_id, sync_id FROM snapshots "
                                f"WHERE agent_id = ? AND source_id IN ({placeholders})",
                                (agent_id, *batch),
                            ).fetchall()
                        )
            return tuple(
                CatalogSnapshotRef(
                    agent_id=row_agent_id,
                    source_id=source_id,
                    sync_id=sync_id,
                )
                for row_agent_id, source_id, sync_id in sorted(
                    rows,
                    key=lambda item: (item[1], item[2]),
                )
            )

        return await asyncio.to_thread(read)

    async def load_current_snapshot(
        self,
        ref: CatalogSnapshotRef,
    ) -> SourceCatalogSnapshot | None:
        if not isinstance(ref, CatalogSnapshotRef):
            raise TypeError("ref must be a CatalogSnapshotRef")
        key = (ref.agent_id, ref.source_id, ref.sync_id)

        async with self._decoded_catalog_snapshot_lock:
            cached = self._decoded_catalog_snapshots.get(key)
            if cached is not None:
                current_sync_id = await asyncio.to_thread(
                    _current_snapshot_sync_id,
                    self.path,
                    ref.agent_id,
                    ref.source_id,
                )
                if current_sync_id != ref.sync_id:
                    self._decoded_catalog_snapshots.pop(key, None)
                    return None
                return cached

            row = await asyncio.to_thread(
                _current_snapshot_row,
                self.path,
                ref.agent_id,
                ref.source_id,
            )
            if row is None or row[0] != ref.sync_id:
                return None
            snapshot = await asyncio.to_thread(_decode_catalog_snapshot, row[1])
            if (
                snapshot.sync.agent_id != ref.agent_id
                or snapshot.sync.source_id != ref.source_id
                or snapshot.sync.id != ref.sync_id
            ):
                raise CatalogStoreError(
                    "stored catalog snapshot does not match its exact reference"
                )
            current_sync_id = await asyncio.to_thread(
                _current_snapshot_sync_id,
                self.path,
                ref.agent_id,
                ref.source_id,
            )
            if current_sync_id != ref.sync_id:
                return None
            self._publish_decoded_catalog_snapshot(snapshot)
            return snapshot

    async def load_sync(self, agent_id: str, sync_id: str) -> CatalogSync | None:
        def read() -> CatalogSync | None:
            with _connect(self.path) as connection:
                row = connection.execute(
                    "SELECT data FROM syncs WHERE agent_id = ? AND id = ?",
                    (agent_id, sync_id),
                ).fetchone()
            return None if row is None else decode_catalog_sync(row[0])

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
        if not active_source_ids:
            return CatalogSummary(
                active_source_count=0,
                resource_count=0,
                relationship_count=0,
                latest_successful_sync_completed_at=None,
                is_empty=True,
            )
        snapshots = await self._snapshots(agent_id, active_source_ids)
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
            for snapshot in await self._snapshots(
                agent_id,
                () if source_id is None else (source_id,),
            )
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
            return tuple(decode_semantic_annotation(data) for (data,) in rows)

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
            return None if row is None else decode_semantic_annotation(row[0])

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
                    row is None or decode_semantic_annotation(row[0]) != annotation
                )
                connection.execute(
                    """INSERT INTO semantic_annotations(agent_id, id, data)
                       VALUES (?, ?, ?)
                       ON CONFLICT(agent_id, id) DO UPDATE SET data = excluded.data""",
                    (agent_id, annotation.id, encode_semantic_annotation(annotation)),
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
                        result = decode_loop_exit(result_data)
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
                                run=decode_run_input(input_data),
                                messages=tuple(
                                    decode_message(data) for (data,) in message_rows
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
            return None if row is None else decode_learning_candidate(row[0])

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
                    (agent_id, candidate.id, encode_learning_candidate(candidate)),
                )
                by_id[candidate.id] = candidate
                identities.add(candidate.candidate_identity_sha256)
                inserted.append(candidate)
            connection.execute(
                """INSERT INTO metadata(key, data) VALUES (?, ?)
                   ON CONFLICT(key) DO UPDATE SET data = excluded.data""",
                (
                    _learning_review_stamps_key(agent_id),
                    encode_review_stamps(tuple(merged_stamps)),
                ),
            )
            return tuple(inserted)

        return await _run_cancellation_safe_transaction(self.path, write)

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
                (encode_learning_candidate(candidate), agent_id, candidate.id),
            )
            return candidate

        return await _run_cancellation_safe_transaction(self.path, write)

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
            rejected = replace(
                current,
                status=LearningCandidateStatus.REJECTED,
                updated_at=rejected_at,
                rejection_reason=reason,
            )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (encode_learning_candidate(rejected), agent_id, candidate_id),
            )
            return rejected

        return await _run_cancellation_safe_transaction(self.path, write)

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
            accepted = replace(
                current,
                status=LearningCandidateStatus.ACCEPTED,
                updated_at=accepted_at,
            )
            connection.execute(
                """UPDATE learning_candidates SET data = ?
                   WHERE agent_id = ? AND id = ?""",
                (encode_learning_candidate(accepted), agent_id, candidate_id),
            )
            return accepted

        return await _run_cancellation_safe_transaction(self.path, write)

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

        return await _run_cancellation_safe_transaction(self.path, write)

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
                            encode_run_input(run),
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
                    (run_id, int(row[0]), encode_message(message)),
                )

        await asyncio.to_thread(write)

    async def finish(self, result: LoopExit) -> None:
        if result.kind is LoopExitKind.COMPLETED:
            raise ValueError("completed runs require atomic transcript completion")

        def write() -> None:
            with _connect(self.path) as connection:
                cursor = connection.execute(
                    """UPDATE runs SET result = ?
                       WHERE id = ? AND conversation_id = ? AND result IS NULL""",
                    (
                        encode_loop_exit(result),
                        result.run_id,
                        result.conversation_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise KeyError(f"unknown run: {result.run_id}")

        await asyncio.to_thread(write)

    async def complete(
        self,
        result: LoopExit,
        final_message: CanonicalMessage,
    ) -> None:
        """Atomically append final assistant text and terminal run state."""

        if result.kind is not LoopExitKind.COMPLETED:
            raise ValueError("atomic transcript completion requires a completed exit")
        if final_message.role is not MessageRole.ASSISTANT:
            raise ValueError("atomic completion requires an assistant message")

        def write() -> None:
            with _connect(self.path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                run_row = connection.execute(
                    """SELECT input, result FROM runs
                       WHERE id = ? AND conversation_id = ?""",
                    (result.run_id, result.conversation_id),
                ).fetchone()
                if run_row is None:
                    raise KeyError(f"unknown run: {result.run_id}")
                if run_row[1] is not None:
                    raise ValueError(f"run is already terminal: {result.run_id}")
                rows = connection.execute(
                    """SELECT data FROM messages
                       WHERE run_id = ? ORDER BY position""",
                    (result.run_id,),
                ).fetchall()
                transcript = Transcript(
                    run=decode_run_input(run_row[0]),
                    messages=(
                        *(decode_message(row[0]) for row in rows),
                        final_message,
                    ),
                )
                validate_completed_transcript(transcript, result)
                connection.execute(
                    """INSERT INTO messages(run_id, position, data)
                       VALUES (?, ?, ?)""",
                    (result.run_id, len(rows), encode_message(final_message)),
                )
                cursor = connection.execute(
                    """UPDATE runs SET result = ?
                       WHERE id = ? AND conversation_id = ? AND result IS NULL""",
                    (
                        encode_loop_exit(result),
                        result.run_id,
                        result.conversation_id,
                    ),
                )
                if cursor.rowcount != 1:
                    raise RuntimeError("run changed during atomic completion")

        await asyncio.to_thread(write)

    async def recover_unfinished_runs(
        self,
        agent_id: str,
        *,
        created_at: datetime,
    ) -> tuple[LoopExit, ...]:
        """Terminalize runs left unfinished by a previously admitted host."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")

        def write() -> tuple[LoopExit, ...]:
            with _connect(self.path) as connection:
                rows = connection.execute(
                    """SELECT id, conversation_id
                       FROM runs
                       WHERE agent_id = ? AND result IS NULL
                       ORDER BY conversation_id, turn_index""",
                    (agent_id,),
                ).fetchall()
                recovered: list[LoopExit] = []
                for run_id, conversation_id in rows:
                    messages = connection.execute(
                        "SELECT data FROM messages WHERE run_id = ? ORDER BY position",
                        (run_id,),
                    ).fetchall()
                    steps = sum(
                        decode_message(row[0]).role is MessageRole.ASSISTANT
                        for row in messages
                    )
                    result = LoopExit(
                        run_id=run_id,
                        conversation_id=conversation_id,
                        kind=LoopExitKind.INTERRUPTED,
                        reason="previous_process_terminated",
                        steps=steps,
                        created_at=created_at,
                    )
                    cursor = connection.execute(
                        "UPDATE runs SET result = ? WHERE id = ? AND result IS NULL",
                        (encode_loop_exit(result), run_id),
                    )
                    if cursor.rowcount != 1:
                        raise RuntimeError("unfinished run changed during recovery")
                    recovered.append(result)
                return tuple(recovered)

        worker = asyncio.create_task(asyncio.to_thread(write))
        cancelled = False
        while not worker.done():
            try:
                await asyncio.shield(worker)
            except asyncio.CancelledError:
                cancelled = True
        recovered = worker.result()
        if cancelled:
            raise asyncio.CancelledError
        return recovered

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
                run=decode_run_input(run_row[0]),
                messages=tuple(decode_message(row[0]) for row in rows),
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
            return None if row[0] is None else decode_loop_exit(row[0])

        return await asyncio.to_thread(read)

    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
    ) -> tuple[ArtifactRef, ...]:
        """Derive current reachable refs from persisted agent-scoped tool messages."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")
        if run_id is not None and (not isinstance(run_id, str) or not run_id):
            raise ValueError("run_id must be non-empty text or None")
        if conversation_id is not None and (
            not isinstance(conversation_id, str) or not conversation_id
        ):
            raise ValueError("conversation_id must be non-empty text or None")

        def read() -> tuple[ArtifactRef, ...]:
            clauses = ["r.agent_id = ?"]
            values: list[object] = [agent_id]
            if run_id is not None:
                clauses.append("r.id = ?")
                values.append(run_id)
            if conversation_id is not None:
                clauses.append("r.conversation_id = ?")
                values.append(conversation_id)
            where = " AND ".join(clauses)
            with _connect_read_only(self.path) as connection:
                rows = connection.execute(
                    f"""SELECT r.id, r.conversation_id, m.data
                        FROM runs AS r
                        JOIN messages AS m ON m.run_id = r.id
                        WHERE {where}
                        ORDER BY r.id, m.position""",
                    tuple(values),
                ).fetchall()
            refs: dict[str, ArtifactRef] = {}
            for stored_run_id, stored_conversation_id, data in rows:
                message = decode_message(data)
                if message.role is not MessageRole.TOOL:
                    continue
                for block in message.content:
                    if not isinstance(block, ToolResultBlock) or block.is_error:
                        continue
                    value = block.output.get("artifact")
                    if not isinstance(value, Mapping):
                        continue
                    try:
                        ref = artifact_ref_from_mapping(value)
                    except (TypeError, ValueError) as error:
                        raise RuntimeError(
                            "stored artifact reference is invalid"
                        ) from error
                    if (
                        ref.run_id != stored_run_id
                        or ref.conversation_id != stored_conversation_id
                        or ref.call_id != block.call_id
                    ):
                        raise RuntimeError(
                            "stored artifact reference identity does not match its run"
                        )
                    existing = refs.get(ref.artifact_id)
                    if existing is not None and existing != ref:
                        raise RuntimeError("stored artifact identity is ambiguous")
                    refs[ref.artifact_id] = ref
            return tuple(
                sorted(
                    refs.values(), key=lambda item: (item.created_at, item.artifact_id)
                )
            )

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
                        run=decode_run_input(input_data),
                        messages=tuple(
                            decode_message(message[0]) for message in message_rows
                        ),
                    )
                    result = (
                        None if result_data is None else decode_loop_exit(result_data)
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
            run = decode_run_input(row[0])
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
                    result = decode_loop_exit(result_data)
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
                                run=decode_run_input(input_data),
                                messages=tuple(
                                    decode_message(message[0])
                                    for message in message_rows
                                ),
                            ),
                            result=result,
                        )
                    )
            records.reverse()
            return exists, tuple(records), older_completed_exists

        return await asyncio.to_thread(read)

    async def _snapshots(
        self,
        agent_id: str,
        source_ids: tuple[str, ...] = (),
    ) -> tuple[SourceCatalogSnapshot, ...]:
        for attempt in range(2):
            refs = await self.list_current_snapshot_refs(agent_id, source_ids)
            snapshots: list[SourceCatalogSnapshot] = []
            generation_changed = False
            for ref in refs:
                snapshot = await self.load_current_snapshot(ref)
                if snapshot is None:
                    generation_changed = True
                    break
                snapshots.append(snapshot)
            if (
                not generation_changed
                and refs
                == await self.list_current_snapshot_refs(
                    agent_id,
                    source_ids,
                )
            ):
                return tuple(snapshots)
            if attempt == 1:
                break
        raise CatalogStoreError("catalog snapshot generation changed repeatedly")

    async def _relationships(self, agent_id: str) -> tuple[CatalogRelationship, ...]:
        return tuple(
            relationship
            for snapshot in await self._snapshots(agent_id)
            for relationship in snapshot.relationships
        )

    def _publish_decoded_catalog_snapshot(
        self,
        snapshot: SourceCatalogSnapshot,
    ) -> None:
        sync = snapshot.sync
        key = (sync.agent_id, sync.source_id, sync.id)
        existing = self._decoded_catalog_snapshots.get(key)
        for cached_key in tuple(self._decoded_catalog_snapshots):
            if cached_key[:2] == key[:2] and cached_key != key:
                del self._decoded_catalog_snapshots[cached_key]
        if existing is None:
            self._decoded_catalog_snapshots[key] = snapshot

    def _evict_decoded_catalog_source(self, agent_id: str, source_id: str) -> None:
        for key in tuple(self._decoded_catalog_snapshots):
            if key[:2] == (agent_id, source_id):
                del self._decoded_catalog_snapshots[key]


def _catalog_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > 512:
        raise ValueError(f"{field_name} exceeds 512 characters")


def _catalog_snapshot_source_ids(source_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(source_ids, tuple):
        raise TypeError("catalog snapshot source_ids must be a tuple")
    for source_id in source_ids:
        _catalog_identifier(source_id, "catalog snapshot source_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("catalog snapshot source_ids cannot contain duplicates")
    return source_ids


def _current_snapshot_sync_id(
    path: Path,
    agent_id: str,
    source_id: str,
) -> str | None:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT sync_id FROM snapshots WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchone()
    return None if row is None else str(row[0])


def _current_snapshot_row(
    path: Path,
    agent_id: str,
    source_id: str,
) -> tuple[str, str] | None:
    with _connect(path) as connection:
        row = connection.execute(
            "SELECT sync_id, data FROM snapshots "
            "WHERE agent_id = ? AND source_id = ?",
            (agent_id, source_id),
        ).fetchone()
    if row is None:
        return None
    return str(row[0]), str(row[1])


def _current_source_snapshot(
    connection: sqlite3.Connection,
    agent_id: str,
    source_id: str,
) -> SourceCatalogSnapshot | None:
    row = connection.execute(
        "SELECT data FROM snapshots WHERE agent_id = ? AND source_id = ?",
        (agent_id, source_id),
    ).fetchone()
    if row is None:
        return None
    snapshot = decode_catalog_snapshot(row[0])
    if snapshot.sync.agent_id != agent_id or snapshot.sync.source_id != source_id:
        raise SourcePermissionStateError("stored catalog snapshot ownership is invalid")
    return snapshot


def _decode_catalog_snapshot(value: str) -> SourceCatalogSnapshot:
    return decode_catalog_snapshot(value)


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
        (annotation_id, decode_semantic_annotation(data), data)
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
    values = tuple(decode_learning_candidate(data) for (data,) in rows)
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
    value = decode_review_stamps(row[0])
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
    candidate = decode_learning_candidate(row[0])
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


async def _run_cancellation_safe_transaction(
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


def _validate_current_records(connection: sqlite3.Connection) -> None:
    identity: AgentIdentity | None = None
    active_sources: dict[str, str] = {}
    for key, data in connection.execute("SELECT key, data FROM metadata"):
        if key == "identity":
            if identity is not None:
                raise ValueError("state contains duplicate agent identity")
            identity = decode_identity(data)
        elif isinstance(key, str) and key.startswith(_ACTIVE_SOURCE_KEY_PREFIX):
            agent_id = key.removeprefix(_ACTIVE_SOURCE_KEY_PREFIX)
            active_sources[agent_id] = decode_identifier(data)
        elif isinstance(key, str) and key.startswith(
            _LEARNING_REVIEW_STAMPS_KEY_PREFIX
        ):
            agent_id = key.removeprefix(_LEARNING_REVIEW_STAMPS_KEY_PREFIX)
            if not agent_id:
                raise ValueError("stored learning review owner is invalid")
            decode_review_stamps(data)
        else:
            raise ValueError("state metadata key is unsupported")

    mcp_binding_counts: dict[str, int] = {}
    for agent_id, binding_id, data in connection.execute(
        "SELECT agent_id, binding_id, data FROM mcp_server_bindings"
    ):
        mcp_binding_counts[agent_id] = mcp_binding_counts.get(agent_id, 0) + 1
        if mcp_binding_counts[agent_id] > MCP_MAX_BINDINGS_PER_AGENT:
            raise ValueError("stored MCP binding count exceeds its fixed bound")
        binding = decode_mcp_binding(
            data,
            agent_id=agent_id,
            binding_id=binding_id,
        )
        if binding.agent_id != agent_id or binding.binding_id != binding_id:
            raise ValueError("stored MCP binding ownership is invalid")
        if identity is not None and binding.agent_id != identity.id:
            raise ValueError("stored MCP binding belongs to another agent")

    sources: dict[tuple[str, str], SourceRegistration] = {}
    for agent_id, source_id, data in connection.execute(
        "SELECT agent_id, id, data FROM sources"
    ):
        registration = decode_source(data)
        if registration.agent_id != agent_id or registration.id != source_id:
            raise ValueError("stored source ownership is invalid")
        sources[(agent_id, source_id)] = registration
    if identity is not None and any(agent_id != identity.id for agent_id, _ in sources):
        raise ValueError("stored source belongs to another agent")
    for agent_id, source_id in active_sources.items():
        active_registration = sources.get((agent_id, source_id))
        if active_registration is None or not active_registration.active:
            raise ValueError("stored active source selection is invalid")

    syncs: dict[tuple[str, str], CatalogSync] = {}
    for agent_id, sync_id, source_id, data in connection.execute(
        "SELECT agent_id, id, source_id, data FROM syncs"
    ):
        sync = decode_catalog_sync(data)
        if (
            sync.agent_id != agent_id
            or sync.id != sync_id
            or sync.source_id != source_id
            or (agent_id, source_id) not in sources
        ):
            raise ValueError("stored catalog sync ownership is invalid")
        syncs[(agent_id, sync_id)] = sync
    for agent_id, source_id, sync_id, data in connection.execute(
        "SELECT agent_id, source_id, sync_id, data FROM snapshots"
    ):
        snapshot = decode_catalog_snapshot(data)
        if (
            snapshot.sync.agent_id != agent_id
            or snapshot.sync.source_id != source_id
            or snapshot.sync.id != sync_id
            or syncs.get((agent_id, sync_id)) != snapshot.sync
        ):
            raise ValueError("stored catalog snapshot ownership is invalid")

    run_ids: set[str] = set()
    message_positions: dict[str, list[int]] = {}
    for (
        run_id,
        agent_id,
        conversation_id,
        turn_index,
        input_data,
        result,
    ) in connection.execute(
        """SELECT id, agent_id, conversation_id, turn_index, input, result
           FROM runs"""
    ):
        run_input = decode_run_input(input_data)
        if (
            run_input.id != run_id
            or run_input.agent_id != agent_id
            or run_input.conversation_id != conversation_id
            or not isinstance(turn_index, int)
            or isinstance(turn_index, bool)
            or turn_index < 0
        ):
            raise ValueError("stored run ownership is invalid")
        if result is not None:
            try:
                exit_record = decode_loop_exit(result)
            except (TypeError, ValueError):
                pass
            else:
                if (
                    exit_record.run_id != run_id
                    or exit_record.conversation_id != conversation_id
                ):
                    raise ValueError("stored run result ownership is invalid")
        run_ids.add(run_id)
    for run_id, position, data in connection.execute(
        "SELECT run_id, position, data FROM messages ORDER BY run_id, position"
    ):
        if run_id not in run_ids:
            raise ValueError("stored message belongs to an unknown run")
        try:
            decode_message(data)
        except (TypeError, ValueError):
            pass
        message_positions.setdefault(run_id, []).append(position)
    if any(
        positions != list(range(len(positions)))
        for positions in message_positions.values()
    ):
        raise ValueError("stored transcript message positions are not contiguous")

    for agent_id, annotation_id, data in connection.execute(
        "SELECT agent_id, id, data FROM semantic_annotations"
    ):
        annotation = decode_semantic_annotation(data)
        if annotation.agent_id != agent_id or annotation.id != annotation_id:
            raise ValueError("stored semantic annotation ownership is invalid")
    for agent_id, candidate_id, data in connection.execute(
        "SELECT agent_id, id, data FROM learning_candidates"
    ):
        candidate = decode_learning_candidate(data)
        if candidate.agent_id != agent_id or candidate.id != candidate_id:
            raise ValueError("stored learning candidate ownership is invalid")


def _logical_state_fingerprint(connection: sqlite3.Connection) -> str:
    objects = tuple(
        connection.execute("""SELECT type, name, tbl_name, sql FROM sqlite_master
               WHERE name NOT LIKE 'sqlite_%'
               ORDER BY type, name""")
    )
    rows = {
        table: tuple(connection.execute(f'SELECT * FROM "{table}" ORDER BY rowid'))
        for table in sorted(table_names(connection))
    }
    material = json.dumps(
        {"objects": objects, "rows": rows},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(material.encode("utf-8")).hexdigest()


def _temporary_state_path(path: Path, label: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        prefix=f".{path.name}.{label}-",
        suffix=".db",
        dir=path.parent,
    )
    os.close(descriptor)
    candidate = Path(raw_path)
    os.chmod(candidate, 0o600)
    return candidate


def _copy_state_database(
    source: Path,
    destination: Path,
    *,
    expected_fingerprint: str,
) -> None:
    with _connect_read_only(source) as source_connection:
        if _logical_state_fingerprint(source_connection) != expected_fingerprint:
            raise RuntimeError("state changed while its upgrade copy was prepared")
        destination_connection = _connect(destination)
        try:
            source_connection.backup(destination_connection)
            destination_connection.commit()
        finally:
            destination_connection.close()
    os.chmod(destination, 0o600)
    with _connect_read_only(destination) as copied_connection:
        if _logical_state_fingerprint(copied_connection) != expected_fingerprint:
            raise RuntimeError("state upgrade copy does not match its source")
        require_healthy(copied_connection)


def _rollback_state_path(
    path: Path,
    *,
    found_revision: str,
    fingerprint: str,
) -> Path:
    revision = re.sub(r"[^A-Za-z0-9._-]+", "-", found_revision).strip("-._")
    if not revision:
        revision = "unknown"
    return path.with_name(f"{path.name}.rollback-{revision[:80]}-{fingerprint[:12]}")


def _fsync_state_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prune_older_rollback_points(path: Path, retained: Path) -> None:
    for candidate in path.parent.glob(f"{path.name}.rollback-*"):
        if candidate == retained:
            continue
        try:
            candidate.unlink()
        except OSError:
            pass


def _upgrade_staged_state(
    path: Path,
    *,
    applied: int | None,
    expected_preledger: PreledgerShape | None,
    source_fingerprint: str,
    found_revision: str,
    upgrade_gate: _UpgradeCommitGate | None,
) -> bool:
    rollback_candidate = _temporary_state_path(path, "rollback")
    staged = _temporary_state_path(path, "upgrade")
    retained_rollback = _rollback_state_path(
        path,
        found_revision=found_revision,
        fingerprint=source_fingerprint,
    )
    published_rollback = False
    activated = False
    try:
        _copy_state_database(
            path,
            rollback_candidate,
            expected_fingerprint=source_fingerprint,
        )
        _copy_state_database(
            rollback_candidate,
            staged,
            expected_fingerprint=source_fingerprint,
        )

        connection = _connect(staged)
        try:
            if upgrade_gate is None:
                connection.execute("BEGIN IMMEDIATE")
            elif not upgrade_gate.start(connection):
                return False
            if applied is None:
                assert expected_preledger is not None
                bridge_preledger(connection, expected_preledger)
                bridged = inspect_journal(connection)
                if bridged >= len(MIGRATIONS):
                    raise RuntimeError("pre-ledger bridge unexpectedly reached current")
                upgrade_journaled(connection, bridged)
            else:
                if inspect_journal(connection) != applied:
                    raise RuntimeError("migration journal changed during admission")
                upgrade_journaled(connection, applied)
            require_schema(connection, CURRENT_TABLES)
            require_healthy(connection)
            if upgrade_gate is None:
                connection.commit()
            elif not upgrade_gate.commit(connection):
                return False
        except BaseException:
            connection.rollback()
            raise
        finally:
            connection.close()

        with _connect_read_only(staged) as staged_connection:
            if inspect_journal(staged_connection) != len(MIGRATIONS):
                raise RuntimeError("staged state did not reach the current revision")
            _validate_current_records(staged_connection)

        os.chmod(staged, 0o600)
        os.chmod(rollback_candidate, 0o600)
        _fsync_state_file(staged)
        _fsync_state_file(rollback_candidate)

        def activate() -> None:
            nonlocal activated, published_rollback
            if retained_rollback.exists():
                with _connect_read_only(retained_rollback) as existing_rollback:
                    if (
                        _logical_state_fingerprint(existing_rollback)
                        != source_fingerprint
                    ):
                        raise RuntimeError(
                            "existing rollback point has conflicting state"
                        )
                rollback_candidate.unlink()
            else:
                os.replace(rollback_candidate, retained_rollback)
                published_rollback = True
            try:
                os.replace(staged, path)
            except BaseException:
                if published_rollback:
                    retained_rollback.unlink(missing_ok=True)
                    published_rollback = False
                raise
            activated = True

        if upgrade_gate is None:
            activate()
        elif not upgrade_gate.activate(activate):
            return False
        _prune_older_rollback_points(path, retained_rollback)
        return True
    finally:
        rollback_candidate.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)
        if published_rollback and not activated:
            retained_rollback.unlink(missing_ok=True)


def _initialize(
    path: Path,
    *,
    upgrade_gate: _UpgradeCommitGate | None = None,
) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        admitted = _admit_existing_state(path, upgrade_gate=upgrade_gate)
        if admitted:
            os.chmod(path, 0o600)
        return admitted
    with _connect(path) as connection:
        create_current(connection)
    os.chmod(path, 0o600)
    return True


def _admit_existing_state(
    path: Path,
    *,
    upgrade_gate: _UpgradeCommitGate | None = None,
) -> bool:
    try:
        with _connect_read_only(path) as connection:
            if "state_migrations" in table_names(connection):
                applied = inspect_journal(connection)
                if applied == len(MIGRATIONS):
                    return True
                expected_preledger = None
                found_revision = MIGRATIONS[applied - 1].migration_id
            else:
                applied = None
                expected_preledger = identify_preledger(connection)
                found_revision = expected_preledger.value
            source_fingerprint = _logical_state_fingerprint(connection)
    except (MigrationJournalNewerError, PreledgerNewerError) as error:
        raise StateCompatibilityError(
            StateCompatibilityCode.NEWER_REVISION,
            path,
            (
                "This local state was created by a newer Daita release. Install "
                "the same or a newer package. No state was changed."
            ),
            current_revision=CURRENT_REVISION,
            found_revision=(
                error.found_revision
                if isinstance(error, MigrationJournalNewerError)
                else "newer-preledger-state"
            ),
        ) from None
    except PreledgerLegacyError:
        raise StateCompatibilityError(
            StateCompatibilityCode.LEGACY,
            path,
            (
                "This agent home belongs to the unsupported pre-1.0 Daita "
                "framework. Keep it intact and use a current agent home; this "
                "release will not overwrite or partially import it."
            ),
            current_revision=CURRENT_REVISION,
            found_revision="pre-1.0-framework",
        ) from None
    except MigrationJournalError as error:
        raise StateCompatibilityError(
            StateCompatibilityCode.REVISION_UNSUPPORTED,
            path,
            (
                "This local state has an unknown, incomplete, reordered, or edited "
                "migration history. No state was changed. Install the matching "
                "Daita release before continuing."
            ),
            current_revision=CURRENT_REVISION,
            found_revision=error.found_revision or "invalid-journal",
        ) from None
    except PreledgerAdmissionError:
        raise _damaged_state_error(path, "unsupported-preledger-state") from None
    except StateCompatibilityError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise _damaged_state_error(path, None) from None

    try:
        return _upgrade_staged_state(
            path,
            applied=applied,
            expected_preledger=expected_preledger,
            source_fingerprint=source_fingerprint,
            found_revision=found_revision,
            upgrade_gate=upgrade_gate,
        )
    except BaseException as error:
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        raise StateCompatibilityError(
            StateCompatibilityCode.UPGRADE_FAILED,
            path,
            (
                "Daita could not update the local state safely. The active "
                "database was not replaced. Reinstall the prior working Daita "
                "package before continuing, then report this upgrade failure."
            ),
            current_revision=CURRENT_REVISION,
            found_revision=found_revision,
        ) from None


def _damaged_state_error(
    path: Path,
    found_revision: str | None,
) -> StateCompatibilityError:
    return StateCompatibilityError(
        StateCompatibilityCode.DAMAGED,
        path,
        (
            "This agent state database is damaged or does not match its declared "
            "Daita revision. No state was changed. Reinstall the matching Daita "
            "release or restore the database through your normal recovery process."
        ),
        current_revision=CURRENT_REVISION,
        found_revision=found_revision,
    )


def _recover_started_database_write_receipts(
    path: Path,
    clock: Callable[[], datetime],
) -> None:
    try:
        with _connect_read_only(path) as connection:
            rows = tuple(
                connection.execute(
                    "SELECT agent_id, id, data FROM database_write_receipts"
                )
            )
        started = tuple(
            (agent_id, receipt_id, receipt)
            for agent_id, receipt_id, data in rows
            if (receipt := decode_receipt(data)).outcome is DatabaseWriteOutcome.STARTED
        )
        if not started:
            return
        completed_at = _database_write_aware(clock(), "receipt recovery completed_at")
        with _connect(path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            for agent_id, receipt_id, receipt in started:
                recovered = receipt.finish(
                    DatabaseWriteOutcome.OUTCOME_UNKNOWN,
                    completed_at=completed_at,
                    affected_rows=None,
                    normalized_error_code="write_outcome_unknown",
                )
                result = connection.execute(
                    """UPDATE database_write_receipts SET data = ?
                       WHERE agent_id = ? AND id = ? AND data = ?""",
                    (
                        encode_receipt(recovered),
                        agent_id,
                        receipt_id,
                        encode_receipt(receipt),
                    ),
                )
                if result.rowcount != 1:
                    raise RuntimeError(
                        "database write receipt changed during startup recovery"
                    )
    except RuntimeError:
        raise
    except (OSError, sqlite3.Error, TypeError, ValueError):
        raise _damaged_state_error(path, CURRENT_REVISION) from None


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


__all__ = [
    "DatabaseWriteOutcome",
    "DatabaseWriteReceipt",
    "DatabaseWriteReceiptConflictError",
    "PostgreSQLUpdateScope",
    "SQLiteStateStore",
    "SourcePermissionStateError",
    "SourceReadMode",
    "SourceReadScope",
    "StateCompatibilityCode",
    "StateCompatibilityError",
    "database_write_receipt_id",
    "postgresql_update_authorization_fingerprint",
]
