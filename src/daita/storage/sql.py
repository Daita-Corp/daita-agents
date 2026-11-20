"""Shared SQL state operations, validation and atomic domain transitions."""

from __future__ import annotations

import asyncio
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import TypeVar, cast

from .._json import FrozenJsonObject
from ..adapters.mcp import (
    MCP_MAX_BINDING_CANONICAL_BYTES,
    MCP_MAX_BINDINGS_PER_AGENT,
    MCPServerBinding,
)
from ..adapters.models import SourceRegistration
from ..artifacts.models import (
    MAX_ARTIFACTS_PER_AGENT,
    ArtifactRecord,
    ArtifactRef,
    ArtifactState,
)
from ..capabilities import (
    EffectEvidenceBasis,
    EffectObservation,
    ExecutionScope,
    ExecutionScopeKind,
)
from ..catalog.models import (
    CATALOG_MAX_LINKS,
    CatalogChange,
    CatalogCommitSummary,
    CatalogEntry,
    CatalogFacet,
    CatalogLink,
    CatalogRelationship,
    CatalogResource,
    CatalogResourceRevision,
    CatalogSearchRow,
    CatalogSourceState,
    CatalogSummary,
    CatalogSync,
    RelationshipKind,
    ResourceBatch,
    ResourceKind,
)
from ..catalog.protocols import CatalogStoreError
from ..distribution.models import (
    MAX_DELIVERIES_PER_AGENT,
    MAX_DELIVERY_LIST_PAGE_SIZE,
    Delivery,
    DeliveryState,
    DeliverySubjectKind,
    OutcomeArtifactReference,
)
from ..identity import AgentIdentity, AgentIdentityConflictError
from ..jobs.models import MAX_JOB_LIST_PAGE_SIZE, Job, JobState
from ..llm.models import (
    CanonicalMessage,
    MessageRole,
)
from ..loop.analysis import AnalysisEvidence
from ..loop.models import (
    ConversationRun,
    LoopExit,
    LoopExitKind,
    RunInput,
    Transcript,
    validate_completed_transcript,
)
from ..routines.models import (
    MAX_ACTIVE_ROUTINES_PER_AGENT,
    MAX_ROUTINE_HISTORY_PAGE_SIZE,
    MAX_ROUTINE_LIST_PAGE_SIZE,
    MAX_SCHEDULED_ROUTINES_PER_AGENT,
    RoutineOccurrence,
    RoutineOccurrenceDisposition,
    RoutineState,
    ScheduledRoutine,
)
from ..routines.schedule import (
    first_slot,
    manual_slot_key,
    validate_schedule,
)
from ..semantics import (
    SemanticAnnotation,
)
from . import (
    catalog_rows,
    sql_analysis as _analysis_store,
    sql_artifacts as _artifact_store,
    sql_conversations as _conversation_store,
    sql_jobs as _job_store,
    sql_mcp as _mcp_store,
    sql_routines as _routine_store,
    sql_semantics as _semantic_store,
    sql_sources as _source_store,
)
from .sql_connection import (
    SQLConnection,
    SQLDatabase,
    required_row,
)
from .sql_routines import (
    _datetime_us,
    _load_routine_occurrence_row,
    _load_routine_row,
    _replace_routine_row,
    _require_effects_unblocked,
)
from .sqlite_codecs import (
    decode_catalog_link,
    decode_catalog_sync,
    decode_delivery,
    decode_identity,
    decode_loop_exit,
    decode_mcp_binding,
    decode_message,
    decode_receipt,
    decode_relational_write_scope,
    decode_routine_occurrence,
    decode_run_input,
    decode_scheduled_routine,
    decode_semantic_annotation,
    decode_source,
    decode_source_configuration_for_deletion,
    decode_source_read_scope,
    encode_catalog_link,
    encode_catalog_sync,
    encode_delivery,
    encode_identity,
    encode_mcp_binding,
    encode_receipt,
    encode_scheduled_routine,
)
from .sqlite_codecs.analysis import decode_analysis_evidence, encode_analysis_evidence
from .sqlite_codecs.jobs import decode_job
from .sqlite_codecs.mcp_bindings import decode_mcp_credential_reference_for_deletion
from .sqlite_records import (
    EffectOutcome,
    EffectReceipt,
    EffectReceiptConflictError,
    EffectResolution,
    EffectResolutionDecision,
    RelationalWriteScope,
    SourceReadMode,
    SourceReadScope,
    effect_receipt_aware as _effect_receipt_aware,
    effect_receipt_text as _effect_receipt_text,
    validate_effect_receipt_id,
)

_T = TypeVar("_T")


def _load_delivery_row(
    connection: SQLConnection,
    agent_id: str,
    delivery_id: str,
) -> tuple[Delivery, str] | None:
    row = connection.execute(
        "SELECT data FROM deliveries WHERE agent_id = ? AND delivery_id = ?",
        (agent_id, delivery_id),
    ).fetchone()
    if row is None:
        return None
    if not isinstance(row[0], str):
        raise RuntimeError("stored delivery payload is invalid")
    return (
        decode_delivery(row[0], agent_id=agent_id, delivery_id=delivery_id),
        row[0],
    )


def _model_sensitivity_rank(value) -> int:
    order = {
        "public": 0,
        "internal": 1,
        "confidential": 2,
        "restricted": 3,
    }
    return order[value.value]


class _CatalogCommitGate:
    """Linearize task cancellation against one state mutation transaction."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancelled = False
        self._started = False

    def start(self, connection: SQLConnection) -> bool:
        with self._lock:
            if self._cancelled:
                return False
        connection.begin(write=True)
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


class SQLStateStore:
    """Shared domain operations over an explicitly supplied ``SQLDatabase``.

    The caller admits and owns the database. This class does not provision it,
    acquire execution ownership or close borrowed database resources. Its close
    method clears local caches after the caller has drained active operations.
    """

    def __init__(
        self, database: SQLDatabase, *, clock: Callable[[], datetime] | None = None
    ) -> None:
        self._database = database
        self._clock = clock or (lambda: datetime.now(UTC))
        self._unreadable: set[tuple[str, str]] = set()
        self._damaged: set[tuple[str, str]] = set()  # (agent_id, row id)
        self._unreadable_lock = threading.Lock()

    async def close(self) -> None:
        with self._unreadable_lock:
            self._unreadable.clear()

    async def recover_started_effect_receipts(
        self, agent_id: str, *, recovered_at: datetime
    ) -> None:
        """Recover abandoned effects only after acquiring exclusive agent ownership.

        Opening a connection is passive. The admitted host calls this once before
        starting any work, while retaining its writer lock through shutdown.
        """
        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        recovered_at = _effect_receipt_aware(recovered_at, "receipt recovery time")
        await run_sql_transaction(
            self._database,
            lambda connection: _recover_started_effect_receipts(
                connection, agent_id, recovered_at
            ),
        )

    async def admit_job(self, job: Job) -> Job:
        if not isinstance(job, Job):
            raise TypeError("job admission requires Job")
        return await run_sql_transaction(
            self._database, lambda connection: _job_store.insert_job(connection, job)
        )

    async def load_job(self, agent_id: str, job_id: str) -> Job | None:
        def read() -> Job | None:
            with _connect_read_only(self._database) as connection:
                return _job_store.load_job(connection, agent_id, job_id)

        return await settled_thread(read)

    async def list_jobs(
        self,
        agent_id: str,
        *,
        states: frozenset[JobState] = frozenset(),
        limit: int = MAX_JOB_LIST_PAGE_SIZE,
    ) -> tuple[Job, ...]:
        if not 1 <= limit <= MAX_JOB_LIST_PAGE_SIZE:
            raise ValueError("job list limit is outside its bound")

        def read() -> tuple[Job, ...]:
            with _connect_read_only(self._database) as connection:
                return _job_store.list_jobs(
                    connection, agent_id, states=states, limit=limit
                )

        return await settled_thread(read)

    async def replace_job(self, job: Job) -> Job:
        if not isinstance(job, Job):
            raise TypeError("job replacement requires Job")
        return await run_sql_transaction(
            self._database, lambda connection: _job_store.replace_job(connection, job)
        )

    async def finish_job(self, job: Job, delivery: Delivery | None) -> Job:
        if not isinstance(job, Job):
            raise TypeError("job finish requires Job")
        return await run_sql_transaction(
            self._database,
            lambda connection: _job_store.finish_job(connection, job, delivery),
        )

    async def job_effect_receipts(
        self, agent_id: str, job_id: str
    ) -> tuple[EffectReceipt, ...]:
        """Every receipt a job's runs reserved, in reservation order."""

        def read() -> tuple[EffectReceipt, ...]:
            with _connect_read_only(self._database) as connection:
                rows = connection.execute(
                    "SELECT data FROM effect_receipts "
                    "WHERE agent_id = ? AND job_id = ? "
                    f"ORDER BY {connection.insertion_order} LIMIT 257",
                    (agent_id, job_id),
                ).fetchall()
            return tuple(decode_receipt(row[0]) for row in rows[:256])

        return await settled_thread(read)

    async def initialize_identity(self, identity: AgentIdentity) -> AgentIdentity:
        def write(connection: SQLConnection) -> AgentIdentity:
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

        return await run_sql_transaction(self._database, write)

    async def load_identity(self) -> AgentIdentity | None:
        def read() -> AgentIdentity | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
            return None if row is None else decode_identity(row[0])

        return await settled_thread(read)

    async def load_deletion_credential_inventory(
        self,
        agent_id: str,
    ) -> tuple[
        AgentIdentity | None,
        tuple[tuple[str, Mapping[str, object]], ...],
        tuple[str, ...],
    ]:
        """Read identity, source configurations and MCP credential candidates."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")

        def read() -> tuple[
            AgentIdentity | None,
            tuple[tuple[str, Mapping[str, object]], ...],
            tuple[str, ...],
        ]:
            with _connect_read_only(self._database) as connection:
                identity_row = connection.execute(
                    "SELECT data FROM metadata WHERE key = 'identity'"
                ).fetchone()
                rows = connection.execute(
                    """SELECT agent_id, id, data FROM sources
                       WHERE agent_id = ? ORDER BY id""",
                    (agent_id,),
                ).fetchall()
                mcp_rows = (
                    connection.execute(
                        "SELECT data FROM mcp_server_bindings WHERE agent_id = ?",
                        (agent_id,),
                    ).fetchall()
                    if connection.table_exists("mcp_server_bindings")
                    else ()
                )
            identity = (
                None if identity_row is None else decode_identity(identity_row[0])
            )
            sources: list[tuple[str, Mapping[str, object]]] = []
            for row_agent_id, source_id, data in rows:
                if not all(
                    isinstance(item, str) for item in (row_agent_id, source_id, data)
                ):
                    raise ValueError("stored source deletion inventory is invalid")
                sources.append(
                    decode_source_configuration_for_deletion(
                        data,
                        agent_id=agent_id,
                        source_id=source_id,
                    )
                )
            mcp_references: list[str] = []
            for (data,) in mcp_rows:
                reference = decode_mcp_credential_reference_for_deletion(data)
                if reference is not None:
                    mcp_references.append(reference)
            return identity, tuple(sources), tuple(mcp_references)

        return await settled_thread(read)

    async def load_mcp_binding(
        self,
        agent_id: str,
        binding_id: str,
    ) -> MCPServerBinding | None:
        def read() -> MCPServerBinding | None:
            with _connect_read_only(self._database) as connection:
                return _mcp_store.load_binding(connection, agent_id, binding_id)

        return await settled_thread(read)

    async def list_mcp_bindings(
        self,
        agent_id: str,
    ) -> tuple[MCPServerBinding, ...]:
        def read() -> tuple[MCPServerBinding, ...]:
            with _connect_read_only(self._database) as connection:
                return _mcp_store.list_bindings(connection, agent_id)

        return await settled_thread(read)

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
        encoded = encode_mcp_binding(binding)
        if len(encoded.encode("utf-8")) > MCP_MAX_BINDING_CANONICAL_BYTES:
            raise ValueError("MCP binding exceeds its byte bound")
        return await run_sql_transaction(
            self._database,
            lambda connection: _mcp_store.store_binding(
                connection, binding, encoded, expected_revision
            ),
        )

    async def update_mcp_discovery(
        self,
        agent_id: str,
        binding_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
    ) -> MCPServerBinding:
        """Replace local hints without changing any execution admission or revision."""

        return await run_sql_transaction(
            self._database,
            lambda connection: _mcp_store.update_discovery(
                connection,
                agent_id,
                binding_id,
                summary=summary,
                when_to_use=when_to_use,
                keywords=keywords,
            ),
        )

    async def update_source_discovery(
        self,
        agent_id: str,
        source_id: str,
        *,
        summary: str,
        when_to_use: str,
        keywords: tuple[str, ...],
    ) -> SourceRegistration:
        """Replace local discovery hints while retaining source identity and scopes."""

        return await run_sql_transaction(
            self._database,
            lambda connection: _source_store.update_discovery(
                connection,
                agent_id,
                source_id,
                summary=summary,
                when_to_use=when_to_use,
                keywords=keywords,
            ),
        )

    async def admit_scheduled_routine(
        self,
        routine: ScheduledRoutine,
        *,
        run_immediately: bool = False,
    ) -> ScheduledRoutine:
        """Atomically admit one pristine routine, its first due slot and, when
        asked, one immediate occurrence."""

        if not isinstance(routine, ScheduledRoutine):
            raise TypeError("routine must be ScheduledRoutine")
        if (
            routine.revision != 1
            or routine.active_occurrence_id is not None
            or routine.last_occurrence_id is not None
            or bool(routine.last_delivery_ids)
            or routine.charged_tokens != 0
            or routine.charged_cost_usd != Decimal("0")
            or routine.attempt_count != 0
            or routine.occurrence_count != 0
            or routine.consecutive_failures != 0
            or routine.state not in {RoutineState.ACTIVE, RoutineState.PAUSED}
        ):
            raise ValueError(
                "new routine must be one pristine active or paused aggregate"
            )
        validate_schedule(routine.schedule)
        next_due = None
        if routine.state is RoutineState.ACTIVE:
            next_due = first_slot(routine.schedule, not_before=routine.created_at)
            if next_due is None:
                raise ValueError("routine_schedule_passed")
        normalized = replace(routine, next_due_at=next_due)
        encoded = encode_scheduled_routine(normalized)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.admit_routine(
                connection,
                normalized,
                encoded,
                clock=self._clock,
                run_immediately=run_immediately,
            ),
        )

    async def load_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
    ) -> ScheduledRoutine | None:
        def read() -> ScheduledRoutine | None:
            with _connect_read_only(self._database) as connection:
                loaded = _load_routine_row(connection, agent_id, routine_id)
            return None if loaded is None else loaded[0]

        return await settled_thread(read)

    async def list_scheduled_routines(
        self,
        agent_id: str,
        *,
        states: frozenset[RoutineState] = frozenset(),
        limit: int = MAX_ROUTINE_LIST_PAGE_SIZE,
    ) -> tuple[ScheduledRoutine, ...]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_ROUTINE_LIST_PAGE_SIZE
        ):
            raise ValueError("routine list limit is outside its bound")
        states = frozenset(states)
        if any(not isinstance(item, RoutineState) for item in states):
            raise TypeError("routine states must contain RoutineState values")

        def read() -> tuple[ScheduledRoutine, ...]:
            with _connect_read_only(self._database) as connection:
                return _routine_store.list_routines(
                    connection, agent_id, states=states, limit=limit
                )

        return await settled_thread(read)

    async def list_routine_occurrences(
        self,
        agent_id: str,
        routine_id: str,
        *,
        limit: int = MAX_ROUTINE_HISTORY_PAGE_SIZE,
    ) -> tuple[RoutineOccurrence, ...]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_ROUTINE_HISTORY_PAGE_SIZE
        ):
            raise ValueError("routine history limit is outside its bound")

        def read() -> tuple[RoutineOccurrence, ...]:
            with _connect_read_only(self._database) as connection:
                return _routine_store.list_occurrences(
                    connection, agent_id, routine_id, limit=limit
                )

        return await settled_thread(read)

    async def list_active_routine_occurrences(
        self,
        agent_id: str,
        *,
        limit: int = MAX_ROUTINE_HISTORY_PAGE_SIZE,
    ) -> tuple[RoutineOccurrence, ...]:
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or not 1 <= limit <= MAX_ROUTINE_HISTORY_PAGE_SIZE
        ):
            raise ValueError("active occurrence limit is outside its bound")

        def read() -> tuple[RoutineOccurrence, ...]:
            with _connect_read_only(self._database) as connection:
                return _routine_store.list_active_occurrences(
                    connection, agent_id, limit=limit
                )

        return await settled_thread(read)

    async def load_routine_occurrences(
        self,
        agent_id: str,
        occurrence_ids: tuple[str, ...],
    ) -> tuple[RoutineOccurrence, ...]:
        if (
            not isinstance(occurrence_ids, tuple)
            or len(occurrence_ids) > MAX_DELIVERY_LIST_PAGE_SIZE
            or any(not isinstance(item, str) or not item for item in occurrence_ids)
        ):
            raise ValueError("occurrence identities are outside their bound")

        def read() -> tuple[RoutineOccurrence, ...]:
            with _connect_read_only(self._database) as connection:
                return _routine_store.load_occurrences(
                    connection, agent_id, occurrence_ids
                )

        return await settled_thread(read)

    async def revise_scheduled_routine(
        self,
        routine: ScheduledRoutine,
        *,
        expected_revision: int,
    ) -> ScheduledRoutine | None:
        """Conditionally persist one complete material routine revision."""

        if not isinstance(routine, ScheduledRoutine):
            raise TypeError("routine must be ScheduledRoutine")
        validate_schedule(routine.schedule)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.revise_routine(
                connection, routine, expected_revision=expected_revision
            ),
        )

    async def transition_scheduled_routine(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        state: RoutineState,
        transitioned_at: datetime,
    ) -> ScheduledRoutine | None:
        if not isinstance(state, RoutineState):
            raise TypeError("routine transition state is invalid")
        _datetime_us(transitioned_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.transition_routine(
                connection,
                agent_id,
                routine_id,
                expected_revision=expected_revision,
                state=state,
                transitioned_at=transitioned_at,
            ),
        )

    async def next_routine_deadline(self, agent_id: str) -> datetime | None:
        def read() -> datetime | None:
            with _connect_read_only(self._database) as connection:
                return _routine_store.next_deadline(connection, agent_id)

        return await settled_thread(read)

    async def load_routine_occurrence(
        self,
        agent_id: str,
        occurrence_id: str,
    ) -> RoutineOccurrence | None:
        def read() -> RoutineOccurrence | None:
            with _connect_read_only(self._database) as connection:
                loaded = _load_routine_occurrence_row(
                    connection,
                    agent_id,
                    occurrence_id,
                )
            return None if loaded is None else loaded[0]

        return await settled_thread(read)

    async def claim_due_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        expected_due_at: datetime,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence | None:
        """Conditionally claim one due canonical slot and reserve its run budget."""

        _datetime_us(expected_due_at)
        _datetime_us(claimed_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.claim_due_occurrence(
                connection,
                agent_id,
                routine_id,
                expected_revision=expected_revision,
                expected_due_at=expected_due_at,
                claimed_at=claimed_at,
                claim_token=claim_token,
            ),
        )

    async def claim_manual_routine_occurrence(
        self,
        agent_id: str,
        routine_id: str,
        *,
        expected_revision: int,
        authorized_control_call_id: str,
        claimed_at: datetime,
        claim_token: str,
    ) -> RoutineOccurrence | None:
        _datetime_us(claimed_at)
        slot = manual_slot_key(
            routine_id,
            expected_revision,
            authorized_control_call_id,
        )
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.claim_manual_occurrence(
                connection,
                agent_id,
                routine_id,
                expected_revision=expected_revision,
                slot=slot,
                claimed_at=claimed_at,
                claim_token=claim_token,
            ),
        )

    async def bind_routine_occurrence_run(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        claim_token: str,
        run_id: str,
        execution_scope: ExecutionScope,
        bound_at: datetime,
    ) -> RoutineOccurrence | None:
        if not isinstance(execution_scope, ExecutionScope):
            raise TypeError("routine execution scope is invalid")
        _datetime_us(bound_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.bind_occurrence_run(
                connection,
                agent_id,
                occurrence_id,
                claim_token=claim_token,
                run_id=run_id,
                execution_scope=execution_scope,
                bound_at=bound_at,
            ),
        )

    async def mark_routine_occurrence_run_terminal(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        run_id: str,
        terminal_at: datetime,
    ) -> RoutineOccurrence | None:
        _datetime_us(terminal_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.mark_occurrence_run_terminal(
                connection,
                agent_id,
                occurrence_id,
                run_id=run_id,
                terminal_at=terminal_at,
            ),
        )

    async def finalize_routine_occurrence(
        self,
        agent_id: str,
        occurrence_id: str,
        *,
        delivery_id: str,
        finalized_at: datetime,
        failure_code: str | None = None,
        artifact_references: tuple[OutcomeArtifactReference, ...] = (),
        outcome_contract_failure_code: str | None = None,
    ) -> tuple[RoutineOccurrence, Delivery | None] | None:
        """Converge one occurrence, routine counters, slot, and logical delivery."""

        _datetime_us(finalized_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.finalize_occurrence(
                connection,
                agent_id,
                occurrence_id,
                delivery_id=delivery_id,
                finalized_at=finalized_at,
                failure_code=failure_code,
                artifact_references=artifact_references,
                outcome_contract_failure_code=outcome_contract_failure_code,
            ),
        )

    async def recover_stale_routine_occurrences(
        self,
        agent_id: str,
        *,
        recovered_at: datetime,
        claim_token_factory: Callable[[str], str],
    ) -> tuple[RoutineOccurrence, ...]:
        """Fence stale claims and converge already-terminal reserved runs."""

        recovered_us = _datetime_us(recovered_at)
        return await run_sql_transaction(
            self._database,
            lambda connection: _routine_store.recover_stale_occurrences(
                connection,
                agent_id,
                recovered_at=recovered_at,
                recovered_us=recovered_us,
                claim_token_factory=claim_token_factory,
            ),
        )

    async def load_delivery(
        self,
        agent_id: str,
        delivery_id: str,
    ) -> Delivery | None:
        def read() -> Delivery | None:
            with _connect_read_only(self._database) as connection:
                loaded = _load_delivery_row(connection, agent_id, delivery_id)
                return None if loaded is None else loaded[0]

        return await settled_thread(read)

    async def list_deliveries(
        self,
        agent_id: str,
        *,
        conversation_id: str | None = None,
        include_acknowledged: bool = False,
        limit: int = MAX_DELIVERY_LIST_PAGE_SIZE,
    ) -> tuple[Delivery, ...]:
        if not 1 <= limit <= MAX_DELIVERY_LIST_PAGE_SIZE:
            raise ValueError("delivery list limit is outside its bound")

        def read() -> tuple[Delivery, ...]:
            clauses = ["agent_id = ?"]
            parameters: list[object] = [agent_id]
            if conversation_id is not None:
                clauses.append("conversation_id = ?")
                parameters.append(conversation_id)
            if not include_acknowledged:
                clauses.append("state != ?")
                parameters.append(DeliveryState.ACKNOWLEDGED.value)
            parameters.append(limit)
            with _connect_read_only(self._database) as connection:
                rows = connection.execute(
                    "SELECT delivery_id, data FROM deliveries WHERE "
                    + " AND ".join(clauses)
                    + " ORDER BY created_at_us DESC, delivery_id DESC LIMIT ?",
                    tuple(parameters),
                ).fetchall()
            return tuple(
                decode_delivery(data, agent_id=agent_id, delivery_id=delivery_id)
                for delivery_id, data in rows
            )

        return await settled_thread(read)

    async def acknowledge_delivery(
        self,
        agent_id: str,
        delivery_id: str,
        *,
        acknowledged_at: datetime,
    ) -> Delivery | None:
        _datetime_us(acknowledged_at)

        def write(connection: SQLConnection) -> Delivery | None:
            loaded = _load_delivery_row(connection, agent_id, delivery_id)
            if loaded is None:
                return None
            current, encoded = loaded
            if current.visibility_state is DeliveryState.ACKNOWLEDGED:
                return current
            updated = replace(
                current,
                visibility_state=DeliveryState.ACKNOWLEDGED,
                updated_at=acknowledged_at,
                acknowledged_at=acknowledged_at,
            )
            result = connection.execute(
                "UPDATE deliveries SET state = ?, data = ? "
                "WHERE agent_id = ? AND delivery_id = ? AND data = ?",
                (
                    updated.visibility_state.value,
                    encode_delivery(updated),
                    agent_id,
                    delivery_id,
                    encoded,
                ),
            )
            if result.rowcount != 1:
                raise RuntimeError("delivery changed during acknowledgment")
            return updated

        return await run_sql_transaction(self._database, write)

    async def register_source(
        self, registration: SourceRegistration
    ) -> SourceRegistration:
        return await run_sql_transaction(
            self._database,
            lambda connection: _source_store.register(connection, registration),
        )

    async def load_source(
        self, agent_id: str, source_id: str
    ) -> SourceRegistration | None:
        def read() -> SourceRegistration | None:
            with _connect_read_only(self._database) as connection:
                return _source_store.load(connection, agent_id, source_id)

        return await settled_thread(read)

    async def list_sources(self, agent_id: str) -> tuple[SourceRegistration, ...]:
        def read() -> tuple[SourceRegistration, ...]:
            with _connect_read_only(self._database) as connection:
                return _source_store.list_sources(connection, agent_id)

        return await settled_thread(read)

    async def load_source_read_scope(
        self,
        agent_id: str,
        source_id: str,
    ) -> SourceReadScope | None:
        """Load one exact active-source scope; invalid state never becomes all."""

        def read() -> SourceReadScope | None:
            with _connect_read_only(self._database) as connection:
                return _source_store.load_read_scope(connection, agent_id, source_id)

        return await settled_thread(read)

    async def list_relational_write_scopes(
        self,
        agent_id: str,
        source_id: str,
    ) -> tuple[RelationalWriteScope, ...]:
        def read() -> tuple[RelationalWriteScope, ...]:
            with _connect_read_only(self._database) as connection:
                return _source_store.list_write_scopes(connection, agent_id, source_id)

        return await settled_thread(read)

    async def replace_source_permission_scopes(
        self,
        read_scope: SourceReadScope,
        update_scopes: tuple[RelationalWriteScope, ...],
    ) -> SourceRegistration:
        """Atomically replace only the two narrow scope families for one source."""

        if not isinstance(read_scope, SourceReadScope):
            raise TypeError("read_scope must be a SourceReadScope")
        if not isinstance(update_scopes, tuple) or any(
            not isinstance(scope, RelationalWriteScope) for scope in update_scopes
        ):
            raise TypeError("update_scopes must be a tuple of RelationalWriteScope")
        if len({scope.resource_id for scope in update_scopes}) != len(update_scopes):
            raise ValueError("update_scopes cannot contain duplicate resources")
        return await run_sql_transaction(
            self._database,
            lambda connection: _source_store.replace_permission_scopes(
                connection, read_scope, update_scopes
            ),
        )

    async def load_effect_receipt(
        self,
        agent_id: str,
        receipt_id: str,
    ) -> EffectReceipt | None:
        _effect_receipt_text(agent_id, "receipt agent_id")
        validate_effect_receipt_id(receipt_id)

        def read() -> EffectReceipt | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    """SELECT data FROM effect_receipts
                       WHERE agent_id = ? AND id = ?""",
                    (agent_id, receipt_id),
                ).fetchone()
            return None if row is None else decode_receipt(row[0])

        return await settled_thread(read)

    async def load_effect_receipt_for_call(
        self,
        agent_id: str,
        run_id: str,
        call_id: str,
    ) -> EffectReceipt | None:
        _effect_receipt_text(agent_id, "receipt agent_id")
        _effect_receipt_text(run_id, "receipt run_id")
        _effect_receipt_text(call_id, "receipt call_id")

        def read() -> EffectReceipt | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    """SELECT data FROM effect_receipts
                       WHERE agent_id = ? AND run_id = ? AND call_id = ?""",
                    (agent_id, run_id, call_id),
                ).fetchone()
            return None if row is None else decode_receipt(row[0])

        return await settled_thread(read)

    async def load_effect_receipt_for_operation(
        self, agent_id: str, operation_key: str
    ) -> EffectReceipt | None:
        _effect_receipt_text(agent_id, "receipt agent")
        if (
            not isinstance(operation_key, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", operation_key) is None
        ):
            raise ValueError("effect operation key is invalid")

        def read() -> EffectReceipt | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    "SELECT data FROM effect_receipts WHERE agent_id = ? AND operation_key = ?",
                    (agent_id, operation_key),
                ).fetchone()
                return None if row is None else decode_receipt(row[0])

        return await settled_thread(read)

    async def list_effect_receipts(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        routine_id: str | None = None,
        unresolved_only: bool = False,
        limit: int = 20,
        offset: int = 0,
        caller_principal_id: str | None = None,
    ) -> tuple[EffectReceipt, ...]:
        _effect_receipt_text(agent_id, "receipt agent")
        if (
            type(limit) is not int
            or not 1 <= limit <= 50
            or type(offset) is not int
            or offset < 0
        ):
            raise ValueError("receipt inspection page is outside its bound")
        for value in (run_id, routine_id):
            if value is not None:
                _effect_receipt_text(value, "receipt inspection identity")
        if caller_principal_id is not None:
            _effect_receipt_text(caller_principal_id, "receipt caller principal")
        if not isinstance(unresolved_only, bool):
            raise TypeError("unresolved_only must be boolean")

        def read() -> tuple[EffectReceipt, ...]:
            with _connect_read_only(self._database) as connection:
                if caller_principal_id is None:
                    rows = connection.execute(
                        "SELECT data FROM effect_receipts WHERE agent_id = ? AND (CAST(? AS TEXT) IS NULL OR run_id = ?) AND (CAST(? AS TEXT) IS NULL OR routine_id = ?) AND (? = 0 OR unresolved = 1) ORDER BY id LIMIT ? OFFSET ?",
                        (
                            agent_id,
                            run_id,
                            run_id,
                            routine_id,
                            routine_id,
                            int(unresolved_only),
                            limit,
                            offset,
                        ),
                    )
                else:
                    rows = connection.execute(
                        "SELECT effect_receipts.data FROM effect_receipts "
                        "JOIN runs ON runs.id = effect_receipts.run_id "
                        "AND runs.agent_id = effect_receipts.agent_id "
                        "WHERE effect_receipts.agent_id = ? "
                        f"AND {connection.caller_expression('runs.input')} = ? "
                        "AND (CAST(? AS TEXT) IS NULL OR effect_receipts.run_id = ?) "
                        "AND (CAST(? AS TEXT) IS NULL OR effect_receipts.routine_id = ?) "
                        "AND (? = 0 OR effect_receipts.unresolved = 1) "
                        "ORDER BY effect_receipts.id LIMIT ? OFFSET ?",
                        (
                            agent_id,
                            caller_principal_id,
                            run_id,
                            run_id,
                            routine_id,
                            routine_id,
                            int(unresolved_only),
                            limit,
                            offset,
                        ),
                    )
                return tuple(decode_receipt(row[0]) for row in rows)

        return await settled_thread(read)

    async def require_effects_unblocked(
        self, agent_id: str, *, run_id: str | None = None, routine_id: str | None = None
    ) -> None:
        def read() -> None:
            with _connect_read_only(self._database) as connection:
                _require_effects_unblocked(
                    connection, agent_id, run_id=run_id, routine_id=routine_id
                )

        await settled_thread(read)

    async def start_effect_receipt(
        self,
        receipt: EffectReceipt,
        *,
        max_receipts_per_run: int = 64,
    ) -> EffectReceipt:
        if (
            not isinstance(receipt, EffectReceipt)
            or receipt.outcome is not EffectOutcome.STARTED
        ):
            raise ValueError("external execution must begin with a started receipt")
        if (
            type(max_receipts_per_run) is not int
            or not 1 <= max_receipts_per_run <= 256
        ):
            raise ValueError("receipt reservation requires a positive run call bound")

        def write(connection: SQLConnection) -> EffectReceipt:
            identity = connection.execute(
                "SELECT data FROM metadata WHERE key = 'identity'"
            ).fetchone()
            run_row = connection.execute(
                "SELECT input, result FROM runs WHERE agent_id = ? AND id = ?",
                (receipt.agent_id, receipt.run_id),
            ).fetchone()
            if (
                identity is None
                or decode_identity(identity[0]).id != receipt.agent_id
                or run_row is None
                or run_row[1] is not None
            ):
                raise EffectReceiptConflictError(
                    "receipt requires its current owned nonterminal run"
                )
            run = decode_run_input(run_row[0])
            _require_effects_unblocked(
                connection,
                receipt.agent_id,
                run_id=receipt.run_id,
                routine_id=receipt.routine_id,
            )
            existing = connection.execute(
                "SELECT id FROM effect_receipts WHERE agent_id = ? AND (id = ? OR (run_id = ? AND call_id = ?) OR operation_key = ?)",
                (
                    receipt.agent_id,
                    receipt.receipt_id,
                    receipt.run_id,
                    receipt.call_id,
                    receipt.operation_key,
                ),
            ).fetchone()
            if existing is not None:
                raise EffectReceiptConflictError(
                    "execution or operation identity already has a receipt"
                )
            reserved_count = required_row(
                connection.execute(
                    "SELECT COUNT(*) FROM effect_receipts WHERE agent_id = ? AND run_id = ?",
                    (receipt.agent_id, receipt.run_id),
                )
            )[0]
            if reserved_count >= max_receipts_per_run:
                raise EffectReceiptConflictError(
                    "the run receipt reservation bound is exhausted"
                )
            scope = run.start.execution_scope if run.start is not None else None
            job_id = (
                scope.job_id
                if scope is not None and scope.scope_kind is ExecutionScopeKind.JOB
                else None
            )
            if receipt.capability_grant_digest is not None:
                raise EffectReceiptConflictError(
                    "effects are approved at the time; a receipt cannot carry a grant"
                )
            if job_id is not None:
                job = _job_store.load_job(connection, receipt.agent_id, job_id)
                if (
                    receipt.routine_id is not None
                    or job is None
                    or job.state is not JobState.RUNNING
                ):
                    raise EffectReceiptConflictError(
                        "job receipt requires its running job"
                    )
            elif scope is not None:
                routine_id = receipt.routine_id
                if routine_id is None:
                    raise EffectReceiptConflictError(
                        "machine receipt lacks routine or job authorization"
                    )
                if (
                    scope.routine_id != receipt.routine_id
                    or scope.routine_revision != receipt.routine_revision
                    or scope.occurrence_id != receipt.occurrence_id
                ):
                    raise EffectReceiptConflictError(
                        "receipt does not match its scope authorization"
                    )
                occurrence = _load_routine_occurrence_row(
                    connection, receipt.agent_id, receipt.occurrence_id or ""
                )
                routine = _load_routine_row(connection, receipt.agent_id, routine_id)
                if (
                    occurrence is None
                    or routine is None
                    or (
                        occurrence[0].reserved_run_id != receipt.run_id
                        or occurrence[0].routine_revision != receipt.routine_revision
                        or occurrence[0].disposition
                        is not RoutineOccurrenceDisposition.RUNNING
                        or occurrence[0].claim_token is None
                        # The lease fences unbound claims. Once bound, the
                        # immutable run/scope and nonterminal run row fence
                        # dispatch; model reasoning may outlast the claim lease.
                        or routine[0].revision != receipt.routine_revision
                        or routine[0].active_occurrence_id != receipt.occurrence_id
                        or routine[0].state is not RoutineState.ACTIVE
                    )
                ):
                    raise EffectReceiptConflictError(
                        "receipt occurrence claim is no longer current"
                    )
            connection.execute(
                "INSERT INTO effect_receipts(agent_id, id, run_id, call_id, operation_key, routine_id, occurrence_id, grant_digest, unresolved, job_id, data) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)",
                (
                    receipt.agent_id,
                    receipt.receipt_id,
                    receipt.run_id,
                    receipt.call_id,
                    receipt.operation_key,
                    receipt.routine_id,
                    receipt.occurrence_id,
                    receipt.capability_grant_digest,
                    job_id,
                    encode_receipt(receipt),
                ),
            )
            return receipt

        return await run_sql_transaction(self._database, write)

    async def finish_effect_receipt(self, receipt: EffectReceipt) -> EffectReceipt:
        if (
            not isinstance(receipt, EffectReceipt)
            or receipt.outcome is EffectOutcome.STARTED
            or receipt.resolution is not None
        ):
            raise ValueError(
                "finish requires one terminal observation without human resolution"
            )

        def write(connection: SQLConnection) -> EffectReceipt:
            row = connection.execute(
                "SELECT data, job_id FROM effect_receipts WHERE agent_id = ? AND id = ?",
                (receipt.agent_id, receipt.receipt_id),
            ).fetchone()
            if row is None:
                raise EffectReceiptConflictError("started receipt does not exist")
            current = decode_receipt(row[0])
            if current.outcome is not EffectOutcome.STARTED:
                if replace(current, resolution=None) == receipt:
                    return current
                raise EffectReceiptConflictError("terminal receipt is immutable")
            if current != receipt.as_started():
                raise EffectReceiptConflictError(
                    "terminal receipt does not match its started identity"
                )
            connection.execute(
                "UPDATE effect_receipts SET data = ?, unresolved = ? WHERE agent_id = ? AND id = ?",
                (
                    encode_receipt(receipt),
                    int(receipt.unresolved),
                    receipt.agent_id,
                    receipt.receipt_id,
                ),
            )
            if receipt.unresolved:
                _pause_effect_routine(
                    connection, receipt, receipt.finished_at or self._clock()
                )
                if row[1] is not None:
                    _job_store.wait_for_uncertain_write(
                        connection,
                        receipt.agent_id,
                        str(row[1]),
                        receipt.finished_at or self._clock(),
                    )
            return receipt

        return await run_sql_transaction(self._database, write)

    async def resolve_effect_receipt(
        self, agent_id: str, resolution: EffectResolution
    ) -> EffectReceipt:
        if not isinstance(resolution, EffectResolution):
            raise TypeError("effect recovery requires an exact human resolution")

        def write(connection: SQLConnection) -> EffectReceipt:
            row = connection.execute(
                "SELECT data, job_id FROM effect_receipts WHERE agent_id = ? AND id = ?",
                (agent_id, resolution.receipt_id),
            ).fetchone()
            if row is None:
                raise EffectReceiptConflictError(
                    "the exact owned receipt is unavailable"
                )
            receipt = decode_receipt(row[0])
            if receipt.receipt_digest != resolution.receipt_digest:
                raise EffectReceiptConflictError(
                    "the receipt observation changed before recovery"
                )
            if receipt.resolution is not None:
                if receipt.resolution == resolution:
                    return receipt
                raise EffectReceiptConflictError(
                    "an existing human resolution is immutable"
                )
            if receipt.outcome is not EffectOutcome.UNCERTAIN:
                raise EffectReceiptConflictError(
                    "only terminal uncertainty requires recovery"
                )
            resolved = replace(receipt, resolution=resolution)
            connection.execute(
                "UPDATE effect_receipts SET data = ?, unresolved = 0 WHERE agent_id = ? AND id = ?",
                (encode_receipt(resolved), agent_id, receipt.receipt_id),
            )
            if receipt.routine_id is not None:
                loaded = _load_routine_row(connection, agent_id, receipt.routine_id)
                if loaded is None:
                    raise EffectReceiptConflictError(
                        "the producing routine is unavailable"
                    )
                routine, data = loaded
                state = (
                    RoutineState.DISABLED
                    if resolution.decision
                    is EffectResolutionDecision.CLOSE_WITHOUT_RETRY
                    else RoutineState.PAUSED
                )
                _replace_routine_row(
                    connection,
                    data,
                    replace(routine, state=state, updated_at=resolution.resolved_at),
                )
            if row[1] is not None:
                _job_store.release_uncertain_write(
                    connection, agent_id, str(row[1]), resolution.resolved_at
                )
            return resolved

        return await run_sql_transaction(self._database, write)

    async def detach_source(
        self, agent_id: str, source_id: str, detached_at: datetime
    ) -> SourceRegistration:
        return await run_sql_transaction(
            self._database,
            lambda connection: _source_store.detach(
                connection, agent_id, source_id, detached_at
            ),
        )

    async def record_sync(self, sync: CatalogSync) -> CatalogSync:
        def write(connection: SQLConnection) -> CatalogSync:
            connection.execute(
                f"""INSERT INTO syncs(agent_id, id, source_id, data)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, id')}) DO UPDATE SET data = excluded.data""",
                (
                    sync.agent_id,
                    sync.id,
                    sync.source_id,
                    encode_catalog_sync(sync),
                ),
            )
            return sync

        return await run_sql_transaction(self._database, write)

    async def commit_catalog(
        self,
        sync: CatalogSync,
        batches: tuple[ResourceBatch, ...],
        *,
        registration: SourceRegistration | None = None,
        full: bool = True,
        probe_tokens: Mapping[str, str] | None = None,
        probed_at: datetime | None = None,
    ) -> CatalogCommitSummary:
        if (probe_tokens is None) != (probed_at is None):
            raise ValueError("probe tokens and probe time come together")
        if registration is not None and (
            registration.agent_id != sync.agent_id or registration.id != sync.source_id
        ):
            raise ValueError("catalog sync and source registration disagree")

        repair = self._damaged_rows(sync.agent_id)

        def write(connection: SQLConnection) -> CatalogCommitSummary:
            if registration is not None:
                _source_store.attach_registration(connection, registration)
            return catalog_rows.commit_pass(
                connection,
                sync,
                batches,
                full=full,
                note=self._note_for(sync.agent_id),
                repair=repair,
                probe_tokens=probe_tokens,
                probed_at=probed_at,
            )

        summary = await run_sql_transaction(
            self._database, write, commit=_commit_catalog_transaction
        )
        self._repaired(sync.agent_id, repair)
        return summary

    async def save_source_state(self, state: CatalogSourceState) -> None:
        await run_sql_transaction(
            self._database, lambda c: catalog_rows.save_state(c, state)
        )

    async def catalog_detail_counts(
        self, agent_id: str, source_id: str
    ) -> tuple[int, int]:
        return await self._catalog_read(
            agent_id,
            lambda c, _: catalog_rows.detail_counts(c, agent_id, source_id),
        )

    async def commit_source_edit(
        self,
        sync: CatalogSync,
        batches: tuple[ResourceBatch, ...],
        *,
        registration: SourceRegistration,
        replaced_source_id: str,
        replaced_at: datetime,
        read_scope: SourceReadScope,
    ) -> CatalogCommitSummary:
        """Atomically hand one active source connection to discovered truth."""

        if not isinstance(registration, SourceRegistration) or not registration.active:
            raise ValueError("source edit requires an active replacement registration")
        if not isinstance(replaced_source_id, str) or not replaced_source_id:
            raise ValueError("replaced_source_id must be a non-empty string")
        if registration.agent_id != sync.agent_id or registration.id != sync.source_id:
            raise ValueError("catalog sync and replacement registration disagree")
        if (
            read_scope.agent_id != registration.agent_id
            or read_scope.source_id != registration.id
        ):
            raise ValueError("replacement read scope belongs to another source")
        pass_resource_ids = {
            entry.resource.id for batch in batches for entry in batch.entries
        }
        if (
            read_scope.mode is SourceReadMode.SELECTED
            and not set(read_scope.resource_ids) <= pass_resource_ids
        ):
            raise ValueError("replacement read scope is outside discovered catalog")

        repair = self._damaged_rows(sync.agent_id)

        def write(connection: SQLConnection) -> CatalogCommitSummary:
            _source_store.replace_for_edit(
                connection, registration, replaced_source_id, replaced_at, read_scope
            )
            return catalog_rows.commit_pass(
                connection,
                sync,
                batches,
                full=True,
                note=self._note_for(sync.agent_id),
                repair=repair,
            )

        summary = await run_sql_transaction(
            self._database, write, commit=_commit_catalog_transaction
        )
        self._repaired(sync.agent_id, repair)
        return summary

    async def take_unreadable_catalog_sources(self, agent_id: str) -> tuple[str, ...]:
        with self._unreadable_lock:
            taken = sorted(s for a, s in self._unreadable if a == agent_id)
            self._unreadable -= {(agent_id, s) for s in taken}
        return tuple(taken)

    async def _catalog_read(
        self, agent_id: str, read: Callable[[SQLConnection, catalog_rows.Note], _T]
    ) -> _T:
        _catalog_identifier(agent_id, "catalog agent_id")

        def run() -> _T:
            with _connect_read_only(self._database) as connection:
                return read(connection, self._note_for(agent_id))

        return await settled_thread(run)

    def _note_for(self, agent_id: str) -> catalog_rows.Note:
        return lambda source_id, row_id: self._note(agent_id, source_id, row_id)

    def _note(self, agent_id: str, source_id: str, row_id: str = "") -> None:
        with self._unreadable_lock:
            self._unreadable.add((agent_id, source_id))
            if row_id:
                self._damaged.add((agent_id, row_id))

    def _damaged_rows(self, agent_id: str) -> frozenset[str]:
        with self._unreadable_lock:
            return frozenset(row for agent, row in self._damaged if agent == agent_id)

    def _repaired(self, agent_id: str, rows: frozenset[str]) -> None:
        with self._unreadable_lock:
            self._damaged -= {(agent_id, row) for row in rows}

    async def current_syncs(
        self, agent_id: str, source_ids: tuple[str, ...]
    ) -> tuple[CatalogSync, ...]:
        selected = _catalog_source_ids(source_ids)
        return await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.current_syncs(c, agent_id, selected, note),
        )

    async def load_source_state(
        self, agent_id: str, source_id: str
    ) -> CatalogSourceState | None:
        return await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.source_state(c, agent_id, source_id, note),
        )

    async def list_catalog_changes(
        self, agent_id: str, source_id: str, *, limit: int = 50
    ) -> tuple[CatalogChange, ...]:
        if not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500")
        return await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.list_changes(
                c, agent_id, source_id, limit, note
            ),
        )

    async def catalog_structure_changed(
        self, agent_id: str, source_id: str, resource_id: str, since_revision: str
    ) -> bool:
        return await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.structure_changed(
                c, agent_id, source_id, resource_id, since_revision
            ),
        )

    async def newest_catalog_changes(
        self, agent_id: str, resource_ids: tuple[str, ...]
    ) -> dict[str, tuple[str, datetime]]:
        """Each resource's newest change as (native identity, changed_at)."""

        return await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.newest_changes(c, agent_id, resource_ids),
        )

    async def tool_sql_reads(
        self, agent_id: str, tool_name: str, since: datetime
    ) -> tuple[tuple[datetime, str, str], ...]:
        """(run time, source ID, SQL) of this tool's calls since ``since``, newest first."""

        def read() -> tuple[tuple[datetime, str, str], ...]:
            with _connect_read_only(self._database) as connection:
                runs: list[tuple[datetime, str]] = []
                for run_id, data in connection.execute(
                    "SELECT id, input FROM runs WHERE agent_id = ?", (agent_id,)
                ):
                    try:
                        created_at = decode_run_input(data).created_at
                    except (TypeError, ValueError, KeyError):
                        continue
                    if created_at >= since:
                        runs.append((created_at, run_id))
                # shortcut: newest 2,000 runs bound the scan; raise it if counts prove too low.
                runs = sorted(runs, reverse=True)[:2_000]
                times = {run_id: created_at for created_at, run_id in runs}
                ids = list(times)
                reads: list[tuple[datetime, str, str]] = []
                for start in range(0, len(ids), 500):
                    chunk = ids[start : start + 500]
                    for run_id, data in connection.execute(
                        "SELECT run_id, data FROM messages WHERE run_id IN ("
                        + ", ".join("?" * len(chunk))
                        + ")",
                        chunk,
                    ):
                        if tool_name not in data:
                            continue
                        try:
                            calls = decode_message(data).tool_calls
                        except (TypeError, ValueError, KeyError):
                            continue
                        for call in calls:
                            if call.name == tool_name:
                                reads.extend(
                                    (times[run_id], source_id, sql)
                                    for source_id, sql in _call_statements(
                                        call.arguments
                                    )
                                )
                return tuple(sorted(reads, key=lambda item: item[0], reverse=True))

        return await settled_thread(read)

    async def load_sync(self, agent_id: str, sync_id: str) -> CatalogSync | None:
        def read() -> CatalogSync | None:
            with _connect_read_only(self._database) as connection:
                row = connection.execute(
                    "SELECT data FROM syncs WHERE agent_id = ? AND id = ?",
                    (agent_id, sync_id),
                ).fetchone()
            if row is None:
                return None
            try:
                return decode_catalog_sync(row[0])
            except (TypeError, ValueError):
                raise CatalogStoreError("stored catalog sync is undecodable") from None

        return await settled_thread(read)

    async def summarize_catalog(
        self,
        agent_id: str,
        active_source_ids: tuple[str, ...],
    ) -> CatalogSummary:
        """Count the catalog of the active sources that have a readable current sync.

        A source whose current sync was not truncated stored exactly the pass it
        counted (every older row was removed), so its sync supplies the counts.
        A truncated sync kept rows it did not restate, so those sources are
        counted in SQL.
        """

        selected = _catalog_source_ids(active_source_ids)
        syncs = await self.current_syncs(agent_id, selected) if selected else ()
        resource_count = sum(s.resource_count for s in syncs if not s.truncated)
        relationship_count = sum(s.relationship_count for s in syncs if not s.truncated)
        counted = [s.source_id for s in syncs if s.truncated]
        if counted:
            more = await self._catalog_read(
                agent_id, lambda c, _: catalog_rows.counts(c, agent_id, counted)
            )
            resource_count += more[0]
            relationship_count += more[1]
        completions = [s.completed_at for s in syncs if s.completed_at is not None]
        return CatalogSummary(
            active_source_count=len(selected),
            resource_count=resource_count,
            relationship_count=relationship_count,
            latest_successful_sync_completed_at=(
                max(completions) if completions else None
            ),
            is_empty=resource_count == 0,
        )

    async def load_entries(
        self, agent_id: str, resource_ids: tuple[str, ...]
    ) -> tuple[CatalogEntry, ...]:
        found = await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.load_entries(c, agent_id, resource_ids, note),
        )
        return tuple(sorted(found, key=lambda item: item.resource.id))

    async def load_entries_by_identity(
        self, agent_id: str, source_id: str, native_identities: tuple[str, ...]
    ) -> tuple[CatalogEntry, ...]:
        found = await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.load_entries_by_identity(
                c, agent_id, source_id, native_identities, note
            ),
        )
        return tuple(sorted(found, key=lambda item: item.resource.id))

    async def list_resource_names(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
        *,
        other_than: ResourceKind | None = None,
        limit: int | None = None,
    ) -> tuple[tuple[str, str, str], ...]:
        selected = _catalog_source_ids(source_ids)
        return tuple(
            await self._catalog_read(
                agent_id,
                lambda c, note: catalog_rows.resource_names(
                    c,
                    agent_id,
                    selected,
                    None if other_than is None else other_than.value,
                    limit,
                ),
            )
        )

    async def load_entry(self, agent_id: str, resource_id: str) -> CatalogEntry | None:
        found = await self.load_entries(agent_id, (resource_id,))
        return found[0] if found else None

    async def list_entries(
        self, agent_id: str, source_id: str
    ) -> tuple[CatalogEntry, ...]:
        return tuple(
            await self._catalog_read(
                agent_id,
                lambda c, note: catalog_rows.list_entries(c, agent_id, source_id, note),
            )
        )

    async def load_resource(
        self, agent_id: str, resource_id: str
    ) -> CatalogResource | None:
        entry = await self.load_entry(agent_id, resource_id)
        return None if entry is None else entry.resource

    async def load_revision(
        self, agent_id: str, resource_id: str, revision: str
    ) -> CatalogResourceRevision | None:
        entry = await self.load_entry(agent_id, resource_id)
        if entry is None or entry.revision.revision != revision:
            return None
        return entry.revision

    async def list_resources(
        self, agent_id: str, source_id: str | None = None
    ) -> tuple[CatalogResource, ...]:
        entries = await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.list_entries(c, agent_id, source_id, note),
        )
        return tuple(item.resource for item in entries)

    async def load_facets(
        self,
        agent_id: str,
        resource_id: str,
        revision: str | None = None,
    ) -> tuple[CatalogFacet, ...]:
        entry = await self.load_entry(agent_id, resource_id)
        if entry is None or (
            revision is not None and entry.resource.current_revision != revision
        ):
            return ()
        return entry.facets

    async def load_incident_relationships(
        self,
        agent_id: str,
        resource_ids: tuple[str, ...],
        *,
        relationship_kinds: tuple[RelationshipKind, ...] = (),
        limit: int = 50,
    ) -> Mapping[str, tuple[CatalogRelationship, ...]]:
        return await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.incident_relationships(
                c, agent_id, resource_ids, relationship_kinds, limit, note
            ),
        )

    async def search_catalog(
        self,
        agent_id: str,
        source_ids: tuple[str, ...],
        *,
        terms: tuple[str, ...] | None,
        anchor_keys: tuple[str, ...] = (),
    ) -> tuple[CatalogSearchRow, ...]:
        selected = _catalog_source_ids(source_ids)
        return await self._catalog_read(
            agent_id,
            lambda c, note: catalog_rows.search(
                c, agent_id, selected, terms, anchor_keys, note
            ),
        )

    async def save_catalog_link(self, link: CatalogLink) -> None:
        """Create or replace one declared link within the per-agent bound."""

        if not isinstance(link, CatalogLink):
            raise TypeError("link must be a CatalogLink")

        def write(connection: SQLConnection) -> None:
            exists = connection.execute(
                "SELECT 1 FROM catalog_links WHERE agent_id = ? AND id = ?",
                (link.agent_id, link.id),
            ).fetchone()
            count = required_row(
                connection.execute(
                    "SELECT COUNT(*) FROM catalog_links WHERE agent_id = ?",
                    (link.agent_id,),
                )
            )[0]
            if exists is None and count >= CATALOG_MAX_LINKS:
                raise CatalogStoreError(
                    f"an agent can declare at most {CATALOG_MAX_LINKS} catalog links"
                )
            connection.execute(
                f"""INSERT INTO catalog_links(agent_id, id, data)
                   VALUES (?, ?, ?)
                   ON CONFLICT({connection.conflict('agent_id, id')}) DO UPDATE SET data = excluded.data""",
                (link.agent_id, link.id, encode_catalog_link(link)),
            )

        await run_sql_transaction(self._database, write)

    async def delete_catalog_link(self, agent_id: str, link_id: str) -> bool:
        def write(connection: SQLConnection) -> bool:
            cursor = connection.execute(
                "DELETE FROM catalog_links WHERE agent_id = ? AND id = ?",
                (agent_id, link_id),
            )
            return cursor.rowcount == 1

        return await run_sql_transaction(self._database, write)

    async def list_catalog_links(self, agent_id: str) -> tuple[CatalogLink, ...]:
        def read() -> tuple[CatalogLink, ...]:
            with _connect_read_only(self._database) as connection:
                rows = connection.execute(
                    "SELECT data FROM catalog_links WHERE agent_id = ? ORDER BY id",
                    (agent_id,),
                ).fetchall()
            if len(rows) > CATALOG_MAX_LINKS:
                raise CatalogStoreError("stored catalog link collection exceeds bound")
            return tuple(decode_catalog_link(data) for (data,) in rows)

        return await settled_thread(read)

    async def list_semantic_annotations(
        self,
        agent_id: str,
    ) -> tuple[SemanticAnnotation, ...]:
        """Return one bounded deterministic agent-isolated semantic collection."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")

        def read() -> tuple[SemanticAnnotation, ...]:
            with _connect_read_only(self._database) as connection:
                return _semantic_store.list_annotations(connection, agent_id)

        return await settled_thread(read)

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
            with _connect_read_only(self._database) as connection:
                return _semantic_store.load_annotation(
                    connection, agent_id, annotation_id
                )

        return await settled_thread(read)

    async def preflight_semantic_save(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        expected_sha256: str | None,
    ) -> FrozenJsonObject:
        """Validate and fingerprint one exact semantic create or replacement."""

        _semantic_store.validate_owner(agent_id, annotation)

        def read() -> FrozenJsonObject:
            with _connect_read_only(self._database) as connection:
                return _semantic_store.save_fingerprint(
                    connection,
                    agent_id,
                    annotation,
                    expected_sha256,
                )

        return await settled_thread(read)

    async def save_semantic_annotation(
        self,
        agent_id: str,
        annotation: SemanticAnnotation,
        *,
        expected_sha256: str | None = None,
    ) -> bool:
        """Atomically create, digest-replace, or digest-supersede one annotation."""

        _semantic_store.validate_owner(agent_id, annotation)
        return await run_sql_transaction(
            self._database,
            lambda connection: _semantic_store.save_annotation(
                connection, agent_id, annotation, expected_sha256
            ),
        )

    async def preflight_semantic_delete(
        self,
        agent_id: str,
        annotation_id: str,
        expected_sha256: str,
    ) -> FrozenJsonObject:
        """Validate and fingerprint one digest-protected semantic deletion."""

        def read() -> FrozenJsonObject:
            with _connect_read_only(self._database) as connection:
                return _semantic_store.delete_fingerprint(
                    connection,
                    agent_id,
                    annotation_id,
                    expected_sha256,
                )

        return await settled_thread(read)

    async def delete_semantic_annotation(
        self,
        agent_id: str,
        annotation_id: str,
        *,
        expected_sha256: str,
    ) -> bool:
        """Atomically delete one annotation only when its rendered digest matches."""

        return await run_sql_transaction(
            self._database,
            lambda connection: _semantic_store.delete_annotation(
                connection, agent_id, annotation_id, expected_sha256
            ),
        )

    _UNSPECIFIED_PREDECESSOR = object()

    async def start(
        self,
        run: RunInput,
        *,
        predecessor=_UNSPECIFIED_PREDECESSOR,
    ) -> Transcript:
        if run.conversation_id is None:
            raise ValueError("run conversation_id must be resolved before persistence")
        predecessor_given = predecessor is not self._UNSPECIFIED_PREDECESSOR
        return await run_sql_transaction(
            self._database,
            lambda connection: _conversation_store.start_run(
                connection,
                run,
                predecessor if predecessor_given else None,
                predecessor_given=predecessor_given,
            ),
        )

    async def append(self, run_id: str, message: CanonicalMessage) -> None:
        def position() -> int:
            with _connect_read_only(self._database) as connection:
                return _conversation_store.next_position(connection, run_id)

        await self.append_at(run_id, await settled_thread(position), message)

    async def append_at(
        self,
        run_id: str,
        position: int,
        message: CanonicalMessage,
    ) -> None:
        await run_sql_transaction(
            self._database,
            lambda connection: _conversation_store.append_message(
                connection, run_id, position, message
            ),
        )

    async def finish(self, result: LoopExit) -> None:
        if result.kind is LoopExitKind.COMPLETED:
            raise ValueError("completed runs require atomic transcript completion")
        await run_sql_transaction(
            self._database,
            lambda connection: _conversation_store.finish_run(connection, result),
        )

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
        await run_sql_transaction(
            self._database,
            lambda connection: _conversation_store.complete_run(
                connection, result, final_message
            ),
        )

    async def recover_unfinished_runs(
        self,
        agent_id: str,
        *,
        created_at: datetime,
    ) -> tuple[LoopExit, ...]:
        """Terminalize runs left unfinished by a previously admitted host."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be non-empty text")
        return await run_sql_transaction(
            self._database,
            lambda connection: _conversation_store.recover_unfinished_runs(
                connection, agent_id, created_at
            ),
        )

    async def load(self, run_id: str) -> Transcript:
        def read() -> Transcript:
            with _connect_read_only(self._database) as connection:
                return _conversation_store.load_transcript(connection, run_id)

        return await settled_thread(read)

    async def result(self, run_id: str) -> LoopExit | None:
        def read() -> LoopExit | None:
            with _connect_read_only(self._database) as connection:
                return _conversation_store.load_result(connection, run_id)

        return await settled_thread(read)

    async def get_artifact_record(self, artifact_id: str) -> ArtifactRecord | None:
        return await _run_read_transaction(
            self._database,
            lambda connection: _artifact_store.get_record(connection, artifact_id),
        )

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
        after: tuple[datetime, str] | None = None,
        newest_first: bool = False,
    ) -> tuple[ArtifactRecord, ...]:
        if not isinstance(newest_first, bool):
            raise ValueError("artifact newest_first must be a boolean")
        if limit is not None and (type(limit) is not int or not 1 <= limit <= 1000):
            raise ValueError("artifact page limit must be between 1 and 1000")
        if type(offset) is not int or offset < 0:
            raise ValueError("artifact offset must be non-negative")
        if after is not None:
            if newest_first:
                raise ValueError("artifact cursor requires creation order")
            if offset or limit is None:
                raise ValueError(
                    "artifact cursor requires a bounded page without offset"
                )
            if (
                not isinstance(after, tuple)
                or len(after) != 2
                or not isinstance(after[0], datetime)
                or after[0].tzinfo is None
                or not isinstance(after[1], str)
                or not after[1]
            ):
                raise ValueError("artifact cursor must contain timestamp and identity")

        return await _run_read_transaction(
            self._database,
            lambda connection: _artifact_store.list_records(
                connection,
                agent_id,
                state=state,
                run_id=run_id,
                conversation_id=conversation_id,
                caller_principal_id=caller_principal_id,
                limit=limit,
                offset=offset,
                after=after,
                newest_first=newest_first,
            ),
        )

    async def list_artifact_refs(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        conversation_id: str | None = None,
        caller_principal_id: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        after: tuple[datetime, str] | None = None,
        newest_first: bool = False,
    ) -> tuple[ArtifactRef, ...]:
        return tuple(
            record.ref
            for record in await self.list_artifact_records(
                agent_id,
                state=ArtifactState.READY,
                run_id=run_id,
                conversation_id=conversation_id,
                caller_principal_id=caller_principal_id,
                limit=limit,
                offset=offset,
                after=after,
                newest_first=newest_first,
            )
        )

    async def begin_artifact_creation(
        self, record: ArtifactRecord, *, reserved: bool = False
    ) -> None:
        if record.state is not ArtifactState.CREATING:
            raise ValueError("new artifact must be creating")
        await run_sql_transaction(
            self._database,
            lambda connection: _artifact_store.begin_creation(
                connection, record, reserved
            ),
        )

    async def transition_artifact(
        self, record: ArtifactRecord, state: ArtifactState
    ) -> ArtifactRecord:
        if not (
            (record.state is ArtifactState.CREATING and state is ArtifactState.READY)
            or (
                record.state is not ArtifactState.DELETING
                and state is ArtifactState.DELETING
            )
        ):
            raise ValueError("artifact lifecycle transition is invalid")
        updated = replace(record, state=state)
        return await run_sql_transaction(
            self._database,
            lambda connection: _artifact_store.transition(connection, record, updated),
        )

    async def finish_artifact_deletion(self, record: ArtifactRecord) -> None:
        if record.state is not ArtifactState.DELETING:
            raise ValueError("artifact deletion must be pending")
        await run_sql_transaction(
            self._database,
            lambda connection: _artifact_store.finish_deletion(connection, record),
        )

    async def conversation_run_page(
        self,
        agent_id: str,
        conversation_id: str,
        *,
        after_turn_index: int = -1,
        limit: int = 100,
    ) -> tuple[ConversationRun, ...]:
        """Read at most 100 turns after an exclusive stable cursor in one snapshot."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("conversation page limit must be between 1 and 100")
        if type(after_turn_index) is not int or after_turn_index < -1:
            raise ValueError("conversation cursor must be at least -1")
        return await _run_read_transaction(
            self._database,
            lambda connection: _conversation_store.conversation_records(
                connection,
                agent_id,
                conversation_id,
                after_turn_index=after_turn_index,
                limit=limit,
            ),
        )

    async def conversation_runs(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> tuple[ConversationRun, ...]:
        """Compatibility full-history read; interactive callers should use pages."""
        return await _run_read_transaction(
            self._database,
            lambda connection: _conversation_store.conversation_records(
                connection, agent_id, conversation_id
            ),
        )

    async def conversation_access(
        self,
        agent_id: str,
        conversation_id: str,
        *,
        caller_principal_id: str | None = None,
    ) -> tuple[bool, bool]:
        """Return existence and whole-conversation caller access without messages."""

        return await _run_read_transaction(
            self._database,
            lambda connection: _conversation_store.conversation_access(
                connection, agent_id, conversation_id, caller_principal_id
            ),
        )

    async def conversation_exists(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> bool:
        """Return one agent-scoped existence fact without loading transcript data."""

        def read() -> bool:
            with _connect_read_only(self._database) as connection:
                return _conversation_store.conversation_exists(
                    connection, agent_id, conversation_id
                )

        return await settled_thread(read)

    async def record_analysis_evidence(
        self, agent_id: str, evidence: AnalysisEvidence
    ) -> None:
        encoded = encode_analysis_evidence(evidence)
        await run_sql_transaction(
            self._database,
            lambda connection: _analysis_store.record_evidence(
                connection, agent_id, evidence, encoded
            ),
        )

    async def load_analysis_evidence(
        self, agent_id: str, run_id: str
    ) -> tuple[AnalysisEvidence, ...]:
        return await _run_read_transaction(
            self._database,
            lambda connection: _analysis_store.load_evidence(
                connection, agent_id, run_id
            ),
        )

    async def pending_analysis_evidence(
        self, agent_id: str, *, after: tuple[str, str] = ("", "")
    ) -> tuple[AnalysisEvidence, ...]:
        """Page candidate evidence, including terminal-run cleanup, before filtering.

        The last returned identity is the next cursor even for wholly closed pages.
        """

        return await _run_read_transaction(
            self._database,
            lambda connection: _analysis_store.pending_evidence(
                connection, agent_id, after
            ),
        )

    async def recover_analysis_evidence(
        self,
        agent_id: str,
        expected: AnalysisEvidence,
        recovery: Mapping[str, object] | None = None,
        *,
        proof_disposed: bool = False,
    ) -> AnalysisEvidence:
        """Append once-only authenticated cleanup facts, without rewriting history.

        Only composition's native recovery calls this operation. Exact encoded CAS
        binds the evidence identity and its retained proof, including terminal runs.
        No native I/O is performed under the transaction.
        """
        before = encode_analysis_evidence(expected)
        facts = dict(expected.facts)
        if recovery is not None:
            if expected.kind != "generation" or facts.get("recovery") is not None:
                raise ValueError("Recovery is once-only native generation evidence")
            if facts["status"] not in {"admitted", "running", "cleanup_failed"}:
                raise ValueError("Recovery requires unresolved generation closure")
            for name in (
                "scratch",
                "scratch_device",
                "scratch_inode",
                "closure_path",
                "closure_device",
                "closure_inode",
                "closure_nonce",
            ):
                if recovery.get(name) != facts.get(name):
                    raise ValueError("Recovery proof identity changed")
            facts["recovery"] = {
                name: recovery[name] for name in ("status", "pid", "usage", "cleanup")
            }
        if proof_disposed:
            proved = facts.get("recovery", facts)
            if expected.kind != "generation" or not isinstance(proved, Mapping):
                raise ValueError("Proof disposal requires generation closure")
            cleanup, usage = proved.get("cleanup"), proved.get("usage")
            if (
                not isinstance(cleanup, Mapping)
                or not isinstance(usage, Mapping)
                or cleanup.get("process_reaped") is not True
                or cleanup.get("scratch_deleted") is not True
                or usage.get("cpu_complete") is not True
            ):
                raise ValueError("Proof disposal requires durable complete closure")
            facts["proof_disposed"] = True
        updated = AnalysisEvidence(
            expected.run_id, expected.evidence_id, expected.kind, facts
        )
        encoded = encode_analysis_evidence(updated)
        await run_sql_transaction(
            self._database,
            lambda connection: _analysis_store.replace_recovered_evidence(
                connection, agent_id, expected, before, encoded, proof_disposed
            ),
        )
        return updated

    async def clear_conversations(self, agent_id: str) -> int:
        """Delete transcripts, keeping runs that active work still needs."""

        if not isinstance(agent_id, str) or not agent_id:
            raise ValueError("agent_id must be a non-empty string")
        return await run_sql_transaction(
            self._database,
            lambda connection: _conversation_store.clear_conversations(
                connection, agent_id
            ),
        )

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
            with _connect_read_only(self._database) as connection:
                return _conversation_store.completed_conversation_tail(
                    connection, agent_id, conversation_id, limit
                )

        return await settled_thread(read)

    async def latest_terminal_conversation_run(
        self,
        agent_id: str,
        conversation_id: str,
    ) -> ConversationRun | None:
        """Return the exact latest terminal revision used for writer CAS binding."""

        def read() -> ConversationRun | None:
            connection = _connect_read_only(self._database)
            try:
                return _conversation_store.latest_terminal_conversation_run(
                    connection, agent_id, conversation_id
                )
            finally:
                connection.close()

        return await settled_thread(read)


def _catalog_identifier(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be non-empty text")
    if value != value.strip():
        raise ValueError(f"{field_name} cannot have surrounding whitespace")
    if len(value) > 512:
        raise ValueError(f"{field_name} exceeds 512 characters")


def _catalog_source_ids(source_ids: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(source_ids, tuple):
        raise TypeError("catalog source_ids must be a tuple")
    for source_id in source_ids:
        _catalog_identifier(source_id, "catalog source_id")
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("catalog source_ids cannot contain duplicates")
    return source_ids


async def settled_thread(
    function: Callable[..., _T],
    /,
    *arguments: object,
) -> _T:
    """Run bounded state work in a thread that settles before cancellation.

    A cancelled ``asyncio.to_thread`` caller returns while its thread keeps a
    database connection open. Waiting for the thread keeps every connection a
    task opened released before that task finishes, so owners that join their
    tasks before closing storage leave no connection behind. The worker's own
    result or failure is discarded when the caller was cancelled.
    """
    worker = asyncio.ensure_future(asyncio.to_thread(function, *arguments))
    cancelled = False
    while not worker.done():
        try:
            await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled = True
    if cancelled:
        if not worker.cancelled():
            worker.exception()  # Retrieve it so the loop does not report it.
        raise asyncio.CancelledError
    return worker.result()


async def run_sql_transaction(
    database: SQLDatabase,
    callback: Callable[[SQLConnection], _T],
    *,
    commit: Callable[[SQLConnection], None] | None = None,
) -> _T:
    """Run one short adapter transaction through the shared cancellation owner.

    Cancellation before admission prevents the write; after admission the worker
    settles exactly once and returns its result or storage failure. The callback
    runs in a worker thread and must perform only bounded database work. No model,
    source, artifact or other external I/O belongs inside this transaction.
    ``commit`` permits an explicit commit boundary; it must never replay work.
    """
    gate = _CatalogCommitGate()
    cancelled_sentinel = object()

    def write() -> _T | object:
        connection = _connect(database)
        try:
            if not gate.start(connection):
                return cancelled_sentinel
            result = callback(connection)
            if commit is None:
                connection.commit()
            else:
                commit(connection)
            return result
        except BaseException as error:
            connection.rollback()
            raise database.normalize_error(error) from None
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
        if result is not cancelled_sentinel:
            raise AssertionError("cancelled state transaction committed")
        raise asyncio.CancelledError
    if result is cancelled_sentinel:
        raise AssertionError("state transaction stopped without cancellation")
    return cast(_T, result)


def _call_statements(arguments: Mapping[str, object]) -> Iterable[tuple[str, str]]:
    """(source ID, SQL text) of a query call; one entry per source it reads."""

    inputs = arguments.get("inputs")
    if isinstance(inputs, (list, tuple)):  # a combined read: SQL per input
        for item in inputs:
            if isinstance(item, Mapping) and isinstance(item.get("sql"), str):
                yield str(item.get("source_id")), str(item["sql"])
        return
    statements = arguments.get("statements")
    parts = (
        [item.get("sql") for item in statements if isinstance(item, Mapping)]
        if isinstance(statements, (list, tuple))
        else [arguments.get("sql")]
    )
    text = "\n".join(part for part in parts if isinstance(part, str))
    if text:
        yield str(arguments.get("source_id")), text


async def _run_read_transaction(
    path: SQLDatabase,
    callback: Callable[[SQLConnection], _T],
    *,
    commit: Callable[[SQLConnection], None] | None = None,
) -> _T:
    def read() -> _T:
        try:
            with _connect_read_only(path) as connection:
                connection.begin()
                return callback(connection)
        except Exception as error:
            raise path.normalize_error(error) from None

    return await settled_thread(read)


def _validate_current_records(connection: SQLConnection) -> AgentIdentity | None:
    _mcp_store.validate_stored_bounds(connection)
    identity: AgentIdentity | None = None
    for key, data in connection.execute("SELECT key, data FROM metadata"):
        if key == "identity":
            if identity is not None:
                raise ValueError("state contains duplicate agent identity")
            identity = decode_identity(data)
        else:
            raise ValueError("state metadata key is unsupported")

    records = tuple(
        _artifact_store.record_from_row(row)
        for row in connection.execute(
            "SELECT artifact_id, agent_id, run_id, conversation_id, caller_principal_id, state, byte_size, created_at_us, data FROM artifacts"
        )
    )
    if len(records) > MAX_ARTIFACTS_PER_AGENT:
        raise ValueError("stored artifact registry exceeds its fixed bound")
    if records and (
        identity is None or any(record.agent_id != identity.id for record in records)
    ):
        raise ValueError("stored artifact belongs to another agent")

    analytical_counts: dict[str, int] = {}
    analytical_bytes: dict[str, int] = {}
    for (
        agent_id,
        run_id,
        evidence_id,
        kind,
        data,
        reserved_bytes,
        run_owner,
    ) in connection.execute(
        "SELECT a.agent_id, a.run_id, a.evidence_id, a.kind, a.data, a.reserved_bytes, r.agent_id "
        "FROM analysis_evidence a LEFT JOIN runs r ON r.id = a.run_id"
    ):
        evidence = decode_analysis_evidence(data)
        if (
            agent_id != run_owner
            or identity is None
            or agent_id != identity.id
            or (evidence.run_id, evidence.evidence_id, evidence.kind)
            != (run_id, evidence_id, kind)
            or type(reserved_bytes) is not int
            or reserved_bytes < len(data.encode())
        ):
            raise ValueError(
                "Stored analytical evidence identity or byte reservation is invalid"
            )
        analytical_counts[run_id] = analytical_counts.get(run_id, 0) + 1
        analytical_bytes[run_id] = analytical_bytes.get(run_id, 0) + reserved_bytes
        if (
            analytical_counts[run_id] > 256
            or analytical_bytes[run_id] > 4 * 1024 * 1024
        ):
            raise ValueError("Stored analytical evidence exceeds its run bounds")

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
    read_scopes: dict[tuple[str, str], SourceReadScope] = {}
    for agent_id, source_id, data in connection.execute(
        "SELECT agent_id, source_id, data FROM source_read_scopes"
    ):
        key = (agent_id, source_id)
        if key in read_scopes:
            raise ValueError("stored source read scope identity is duplicated")
        read_scopes[key] = decode_source_read_scope(
            data,
            agent_id=agent_id,
            source_id=source_id,
        )
    for key, registration in sources.items():
        scope = read_scopes.pop(key, None)
        if registration.active and scope is None:
            raise ValueError("active source is missing its read scope")
        if not registration.active and scope is not None:
            raise ValueError("detached source retains a read scope")
    if read_scopes:
        raise ValueError("stored source read scope is foreign")

    for agent_id, source_id, resource_id, fingerprint, data in connection.execute(
        """SELECT agent_id, source_id, resource_id,
                  authorization_fingerprint, data
           FROM relational_write_scopes"""
    ):
        scope_registration = sources.get((agent_id, source_id))
        if scope_registration is None or not scope_registration.active:
            raise ValueError("stored PostgreSQL update scope is foreign")
        decode_relational_write_scope(
            data,
            agent_id=agent_id,
            source_id=source_id,
            resource_id=resource_id,
            authorization_fingerprint=fingerprint,
        )

    # Catalog rows are not read here: a damaged row is refreshed
    # from its source, so it must not stop the home opening.

    run_ids: set[str] = set()
    run_inputs: dict[str, RunInput] = {}
    run_results: dict[str, LoopExit] = {}
    run_messages: dict[str, list[CanonicalMessage]] = {}
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
        if identity is not None and run_input.agent_id != identity.id:
            raise ValueError("stored run belongs to another agent")
        if result is not None:
            exit_record = decode_loop_exit(result)
            if (
                exit_record.run_id != run_id
                or exit_record.conversation_id != conversation_id
            ):
                raise ValueError("stored run result ownership is invalid")
            run_results[run_id] = exit_record
        run_ids.add(run_id)
        run_inputs[run_id] = run_input
    for run_id, position, data in connection.execute(
        "SELECT run_id, position, data FROM messages ORDER BY run_id, position"
    ):
        if run_id not in run_ids:
            raise ValueError("stored message belongs to an unknown run")
        message = decode_message(data)
        message_positions.setdefault(run_id, []).append(position)
        run_messages.setdefault(run_id, []).append(message)
    if any(
        positions != list(range(len(positions)))
        for positions in message_positions.values()
    ):
        raise ValueError("stored transcript message positions are not contiguous")
    for run_id, result in run_results.items():
        if result.kind is LoopExitKind.COMPLETED:
            validate_completed_transcript(
                Transcript(
                    run=run_inputs[run_id],
                    messages=tuple(run_messages.get(run_id, ())),
                ),
                result,
            )

    for (
        agent_id,
        receipt_id,
        run_id,
        call_id,
        operation_key,
        routine_id,
        occurrence_id,
        grant_digest,
        unresolved,
        data,
    ) in connection.execute(
        """SELECT agent_id, id, run_id, call_id, operation_key, routine_id, occurrence_id, grant_digest, unresolved, data
           FROM effect_receipts"""
    ):
        receipt = decode_receipt(data)
        if (
            receipt.agent_id != agent_id
            or receipt.receipt_id != receipt_id
            or receipt.run_id != run_id
            or receipt.call_id != call_id
            or receipt.operation_key != operation_key
            or receipt.routine_id != routine_id
            or receipt.occurrence_id != occurrence_id
            or receipt.capability_grant_digest != grant_digest
            or int(receipt.unresolved) != unresolved
        ):
            raise ValueError(
                "stored effect receipt identity or indexed state is invalid"
            )
        if identity is not None and receipt.agent_id != identity.id:
            raise ValueError("stored effect receipt belongs to another agent")

    for agent_id, annotation_id, data in connection.execute(
        "SELECT agent_id, id, data FROM semantic_annotations"
    ):
        annotation = decode_semantic_annotation(data)
        if annotation.agent_id != agent_id or annotation.id != annotation_id:
            raise ValueError("stored semantic annotation ownership is invalid")
        if identity is not None and annotation.agent_id != identity.id:
            raise ValueError("stored semantic annotation belongs to another agent")
    for agent_id, link_id, data in connection.execute(
        "SELECT agent_id, id, data FROM catalog_links"
    ):
        link = decode_catalog_link(data)
        if link.agent_id != agent_id or link.id != link_id:
            raise ValueError("stored catalog link ownership is invalid")
        if identity is not None and link.agent_id != identity.id:
            raise ValueError("stored catalog link belongs to another agent")

    jobs: dict[tuple[str, str], Job] = {}
    for agent_id, job_id, conversation_id, state, revision, data in connection.execute(
        "SELECT agent_id, job_id, conversation_id, state, revision, data FROM jobs"
    ):
        job = decode_job(data)
        if (
            job.agent_id != agent_id
            or job.job_id != job_id
            or job.conversation_id != conversation_id
            or job.state.value != state
            or job.revision != revision
        ):
            raise ValueError("stored job projection is invalid")
        if identity is not None and job.agent_id != identity.id:
            raise ValueError("stored job belongs to another agent")
        jobs[(agent_id, job_id)] = job

    routines: dict[tuple[str, str], ScheduledRoutine] = {}
    routine_counts: dict[str, int] = {}
    active_routine_counts: dict[str, int] = {}
    for (
        agent_id,
        routine_id,
        conversation_id,
        state,
        next_due_at_us,
        data,
    ) in connection.execute(
        """SELECT agent_id, routine_id, conversation_id, state,
                  next_due_at_us, data
           FROM scheduled_routines"""
    ):
        routine = decode_scheduled_routine(
            data,
            agent_id=agent_id,
            routine_id=routine_id,
        )
        if (
            routine.conversation_id != conversation_id
            or routine.state.value != state
            or _datetime_us(routine.next_due_at) != next_due_at_us
        ):
            raise ValueError("stored scheduled routine projection is invalid")
        validate_schedule(routine.schedule)
        if identity is not None and routine.agent_id != identity.id:
            raise ValueError("stored scheduled routine belongs to another agent")
        key = (agent_id, routine_id)
        routines[key] = routine
        routine_counts[agent_id] = routine_counts.get(agent_id, 0) + 1
        if routine.state is RoutineState.ACTIVE:
            active_routine_counts[agent_id] = active_routine_counts.get(agent_id, 0) + 1
    if any(
        value > MAX_SCHEDULED_ROUTINES_PER_AGENT for value in routine_counts.values()
    ):
        raise ValueError("stored routine count exceeds its fixed bound")
    if any(
        value > MAX_ACTIVE_ROUTINES_PER_AGENT
        for value in active_routine_counts.values()
    ):
        raise ValueError("stored active routine count exceeds its fixed bound")

    occurrences: dict[tuple[str, str], RoutineOccurrence] = {}
    for (
        agent_id,
        occurrence_id_value,
        routine_id,
        routine_revision,
        slot_key,
        state,
        lease_expires_at_us,
        reserved_run_id,
        data,
    ) in connection.execute(
        """SELECT agent_id, occurrence_id, routine_id, routine_revision,
                  slot_key, state, lease_expires_at_us, reserved_run_id, data
           FROM routine_occurrences"""
    ):
        occurrence = decode_routine_occurrence(
            data,
            agent_id=agent_id,
            occurrence_id=occurrence_id_value,
        )
        owning_routine = routines.get((agent_id, routine_id))
        if (
            owning_routine is None
            or occurrence.routine_id != routine_id
            or occurrence.routine_revision != routine_revision
            or occurrence.routine_revision > owning_routine.revision
            or occurrence.slot_key != slot_key
            or occurrence.disposition.value != state
            or _datetime_us(occurrence.lease_expires_at) != lease_expires_at_us
            or occurrence.reserved_run_id != reserved_run_id
        ):
            raise ValueError("stored routine occurrence projection is invalid")
        occurrences[(agent_id, occurrence_id_value)] = occurrence
    for routine in routines.values():
        if routine.active_occurrence_id is None:
            continue
        active = occurrences.get((routine.agent_id, routine.active_occurrence_id))
        if active is None or active.routine_id != routine.routine_id:
            raise ValueError("stored routine active occurrence is invalid")

    delivery_counts: dict[str, int] = {}
    for (
        agent_id,
        delivery_id,
        conversation_id,
        subject_kind,
        subject_id,
        logical_key,
        target_kind,
        target_fingerprint_value,
        state,
        created_at_us,
        data,
    ) in connection.execute(
        """SELECT agent_id, delivery_id, conversation_id, subject_kind,
                  subject_id, logical_key, target_kind, target_fingerprint,
                  state, created_at_us, data
           FROM deliveries"""
    ):
        delivery = decode_delivery(data, agent_id=agent_id, delivery_id=delivery_id)
        if (
            delivery.conversation_id != conversation_id
            or delivery.subject_kind.value != subject_kind
            or delivery.subject_id != subject_id
            or delivery.logical_key != logical_key
            or target_kind != "conversation_inbox"
            or delivery.target.target_fingerprint != target_fingerprint_value
            or delivery.visibility_state.value != state
            or _datetime_us(delivery.created_at) != created_at_us
        ):
            raise ValueError("stored delivery projection is invalid")
        if identity is not None and delivery.agent_id != identity.id:
            raise ValueError("stored delivery belongs to another agent")
        if delivery.subject_kind is DeliverySubjectKind.ROUTINE_OCCURRENCE:
            producer = occurrences.get((agent_id, subject_id))
            producer_references_delivery = (
                producer is not None and delivery_id in producer.delivery_ids
            )
        elif delivery.subject_kind is DeliverySubjectKind.JOB:
            producing_job = jobs.get((agent_id, subject_id))
            producer_references_delivery = (
                producing_job is not None
                and producing_job.conversation_id == delivery.conversation_id
            )
        else:
            raise ValueError("stored delivery retained a legacy producer")
        if not producer_references_delivery:
            raise ValueError("stored delivery producer reference is invalid")
        delivery_counts[agent_id] = delivery_counts.get(agent_id, 0) + 1
        if delivery_counts[agent_id] > MAX_DELIVERIES_PER_AGENT:
            raise ValueError("stored delivery count exceeds its fixed bound")
    return identity


def _pause_effect_routine(
    connection: SQLConnection, receipt: EffectReceipt, changed_at: datetime
) -> None:
    if receipt.routine_id is None:
        return
    loaded = _load_routine_row(connection, receipt.agent_id, receipt.routine_id)
    if loaded is None:
        raise EffectReceiptConflictError("the producing routine is unavailable")
    routine, encoded = loaded
    _replace_routine_row(
        connection,
        encoded,
        replace(
            routine,
            state=RoutineState.PAUSED,
            updated_at=max(changed_at, routine.updated_at),
        ),
    )


def _recover_started_effect_receipts(
    connection: SQLConnection, agent_id: str, completed_at: datetime
) -> None:
    rows = tuple(
        connection.execute(
            "SELECT agent_id, id, data, job_id FROM effect_receipts WHERE agent_id = ?",
            (agent_id,),
        )
    )
    started = tuple(
        (agent_id, receipt_id, receipt, job_id)
        for (agent_id, receipt_id, data, job_id) in rows
        if (receipt := decode_receipt(data)).outcome is EffectOutcome.STARTED
    )
    for agent_id, receipt_id, receipt, job_id in started:
        recovered = receipt.finish(
            EffectObservation(EffectOutcome.UNCERTAIN, EffectEvidenceBasis.UNKNOWN),
            finished_at=completed_at,
        )
        result = connection.execute(
            """UPDATE effect_receipts SET data = ?
               WHERE agent_id = ? AND id = ? AND data = ?""",
            (
                encode_receipt(recovered),
                agent_id,
                receipt_id,
                encode_receipt(receipt),
            ),
        )
        if result.rowcount != 1:
            raise RuntimeError("effect receipt changed during startup recovery")
        _pause_effect_routine(connection, recovered, completed_at)
        if job_id is not None:
            _job_store.wait_for_uncertain_write(
                connection, agent_id, str(job_id), completed_at
            )


def _commit_catalog_transaction(connection: SQLConnection) -> None:
    connection.commit()


def _connect(database: SQLDatabase) -> SQLConnection:
    return database.connect()


def _connect_read_only(database: SQLDatabase) -> SQLConnection:
    return database.connect(read_only=True)
